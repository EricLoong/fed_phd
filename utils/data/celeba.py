import numpy as np
import pandas as pd
import os
from torchvision.datasets import CelebA
from torch.utils.data import Dataset
from torchvision import transforms

# Custom Dataset to include CelebA attributes
class CelebADataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.celeba = CelebA(root, split=split, download=True, transform=transform)
        attr_file = os.path.join(root, 'celeba', 'list_attr_celeba.txt')

        # Read the .txt file using pandas with the appropriate delimiter
        self.attr = pd.read_csv(attr_file, delim_whitespace=True, header=1)
        self.attr = self.attr.replace(-1, 0)  # Replace -1 with 0 for binary attributes

    def __len__(self):
        return len(self.celeba)

    def __getitem__(self, idx):
        if idx >= len(self.celeba):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self.celeba)}")

        image, _ = self.celeba[idx]
        attributes = self.attr.iloc[idx][['Male', 'Young']]
        class_label = create_classes(attributes)
        return image, class_label

def create_classes(attr):
    # Create a unique class based on binary encoding of the two attributes
    return int(attr['Male']) * 2 + int(attr['Young'])

def partition_data_indices_celeba(datadir, partition, n_nets, n_cls):
    # Ensure n_cls is an integer
    n_cls = int(n_cls)

    # Load CelebA dataset and attributes
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CelebADataset(root=datadir, split='train', transform=transform)
    y_train = dataset.attr[['Male', 'Young']].apply(create_classes, axis=1).values

    # Count samples for each class
    class_counts = np.bincount(y_train, minlength=4)
    total_samples = sum(class_counts)

    for i in range(4):
        print(f"Class {i}: {class_counts[i]} samples")

    # Calculate class proportions and occurrences
    class_proportions = class_counts / total_samples
    class_occurrences = {cls: max(1, int(np.ceil(proportion * n_nets))) for cls, proportion in enumerate(class_proportions)}

    # Initialize the data index map, label distribution map, and class counts per client
    net_dataidx_map = {i: [] for i in range(n_nets)}
    label_distribution = {i: [] for i in range(n_nets)}
    client_class_counts = {i: {cls: 0 for cls in range(4)} for i in range(n_nets)}
    local_number_data = {i: 0 for i in range(n_nets)}

    # Assign labels to clients
    client_assignments = {i: [] for i in range(n_nets)}
    for cls, occurrences in class_occurrences.items():
        assigned_clients = np.random.choice(n_nets, occurrences, replace=False)
        for client_id in assigned_clients:
            client_assignments[client_id].append(cls)
            label_distribution[client_id].append(cls)

    # Ensure each client has exactly n_cls classes
    for client_id in range(n_nets):
        if len(client_assignments[client_id]) < n_cls:
            additional_classes = np.random.choice([cls for cls in range(4) if cls not in client_assignments[client_id]], n_cls - len(client_assignments[client_id]), replace=False)
            client_assignments[client_id].extend(additional_classes)
            label_distribution[client_id].extend(additional_classes)
        elif len(client_assignments[client_id]) > n_cls:
            client_assignments[client_id] = client_assignments[client_id][:n_cls]
            label_distribution[client_id] = label_distribution[client_id][:n_cls]

    # Assign samples to clients based on label distribution
    class_indices = [np.where(y_train == i)[0] for i in range(4)]
    for client_id, classes in client_assignments.items():
        for cls in classes:
            num_samples = len(class_indices[cls]) // class_occurrences[cls]
            assigned_samples = class_indices[cls][:num_samples]
            net_dataidx_map[client_id].extend(assigned_samples)
            client_class_counts[client_id][cls] += len(assigned_samples)
            local_number_data[client_id] += len(assigned_samples)
            class_indices[cls] = class_indices[cls][num_samples:]

    # Assign remaining samples to clients
    for cls, indices in enumerate(class_indices):
        remaining_clients = [client_id for client_id, assigned_classes in client_assignments.items() if cls in assigned_classes]
        idx = 0
        while len(indices) > 0 and idx < len(remaining_clients):
            client_id = remaining_clients[idx]
            net_dataidx_map[client_id].append(indices[0])
            client_class_counts[client_id][cls] += 1
            local_number_data[client_id] += 1
            indices = indices[1:]
            idx += 1
            if idx == len(remaining_clients):
                idx = 0

    # Ensure all data is assigned and there are no out-of-range indices
    #for client_id, indices in net_dataidx_map.items():
    #    out_of_range_indices = [idx for idx in indices if idx < 0 or idx >= len(dataset)]
    #    if out_of_range_indices:
    #        print(f"Client {client_id} has out-of-range indices: {out_of_range_indices}")
    #    assert all(0 <= idx < len(dataset) for idx in indices), f"Client {client_id} has out-of-range indices!"

    # Verify no overlapping indices
    all_indices = [idx for indices in net_dataidx_map.values() for idx in indices]
    if len(all_indices) != len(set(all_indices)):
        print("Overlap detected in data indices!")
        overlapping_indices = set([x for x in all_indices if all_indices.count(x) > 1])
        print(f"Overlapping indices: {overlapping_indices}")
    assert len(all_indices) == len(set(all_indices)), "Overlap detected in data indices!"

    # Print label distribution and number of samples for each client
    for client_id, labels in label_distribution.items():
        print(f"Client {client_id}: {labels}, Total samples: {len(net_dataidx_map[client_id])}")
        print(f"Client {client_id} sample distribution: {client_class_counts[client_id]}")

    return net_dataidx_map, local_number_data, label_distribution

