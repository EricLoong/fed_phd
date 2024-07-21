import numpy as np
import pandas as pd
import os
from math import ceil
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

def adjust_client_distribution(client_counts, total_clients):
    while sum(client_counts) != total_clients:
        diff = sum(client_counts) - total_clients
        if diff > 0:
            max_index = np.argmax(client_counts)
            client_counts[max_index] -= 1
        elif diff < 0:
            min_index = np.argmin(client_counts)
            client_counts[min_index] += 1
    return client_counts

def partition_data_indices_celeba(datadir, partition, n_nets):
    # Load CelebA dataset and attributes
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CelebADataset(root=datadir, split='train', transform=transform)
    y_train = dataset.attr[['Male', 'Young']].apply(create_classes, axis=1).values

    # Count samples for each class
    class_counts = np.bincount(y_train, minlength=4)

    # Initialize the data index map, label distribution map, and local data number map
    net_dataidx_map = {i: [] for i in range(n_nets)}
    label_distribution = {i: [] for i in range(n_nets)}
    local_number_data = {i: 0 for i in range(n_nets)}

    if partition == 'iid':
        all_indices = np.arange(len(y_train))
        np.random.shuffle(all_indices)
        num_samples_per_client = len(y_train) // n_nets
        for client_id in range(n_nets):
            start_idx = client_id * num_samples_per_client
            end_idx = start_idx + num_samples_per_client if client_id < n_nets - 1 else len(y_train)
            assigned_samples = all_indices[start_idx:end_idx]
            net_dataidx_map[client_id].extend(assigned_samples)
            local_number_data[client_id] = len(assigned_samples)
            label_distribution[client_id] = list(np.unique(y_train[assigned_samples]))
    else:  # Non-IID
        # Calculate the number of clients per class
        num_clients_per_class = [ceil(n_nets * class_counts[i] / len(y_train)) for i in range(4)]
        num_clients_per_class = adjust_client_distribution(num_clients_per_class, n_nets)

        client_id = 0
        for cls in range(4):
            class_indices = np.where(y_train == cls)[0]
            np.random.shuffle(class_indices)
            num_clients = num_clients_per_class[cls]
            num_samples_per_client = len(class_indices) // num_clients

            for _ in range(num_clients):
                if client_id >= n_nets:
                    break
                start_idx = _ * num_samples_per_client
                end_idx = start_idx + num_samples_per_client if _ < num_clients - 1 else len(class_indices)
                assigned_samples = class_indices[start_idx:end_idx]

                net_dataidx_map[client_id].extend(assigned_samples)
                label_distribution[client_id].append(cls)
                local_number_data[client_id] = len(assigned_samples)
                client_id += 1

    # Ensure all data is assigned and there are no out-of-range indices
    for client_id, indices in net_dataidx_map.items():
        assert all(0 <= idx < len(dataset) for idx in indices), f"Client {client_id} has out-of-range indices!"

    # Verify no overlapping indices
    all_indices = [idx for indices in net_dataidx_map.values() for idx in indices]
    assert len(all_indices) == len(set(all_indices)), "Overlap detected in data indices!"

    # Print label distribution and number of samples for each client
    for client_id, labels in label_distribution.items():
        print(f"Client {client_id}: {labels}, Total samples: {len(net_dataidx_map[client_id])}")

    return net_dataidx_map, local_number_data, label_distribution
