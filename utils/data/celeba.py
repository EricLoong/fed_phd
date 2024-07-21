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

    # Initialize the data index map and label distribution map
    net_dataidx_map = {i: [] for i in range(n_nets)}
    local_number_data = {i: 0 for i in range(n_nets)}
    label_distribution = {client_id: [] for client_id in range(n_nets)}

    # Gather indices for each class
    class_indices = [np.where(y_train == i)[0] for i in range(4)]  # 4 classes based on attributes

    # Determine the number of samples to be assigned to each client for each class
    samples_per_client_per_class = {i: [] for i in range(4)}
    for cls in range(4):
        num_samples = class_counts[cls]
        samples_per_client = num_samples // n_nets
        remainder_samples = num_samples % n_nets
        for i in range(n_nets):
            if i < remainder_samples:
                samples_per_client_per_class[cls].append(samples_per_client + 1)
            else:
                samples_per_client_per_class[cls].append(samples_per_client)

    # Assign data to clients ensuring no overlap
    for cls in range(4):
        np.random.shuffle(class_indices[cls])
        start_idx = 0
        for i in range(n_nets):
            end_idx = start_idx + samples_per_client_per_class[cls][i]
            net_dataidx_map[i].extend(class_indices[cls][start_idx:end_idx])
            local_number_data[i] += (end_idx - start_idx)
            if cls not in label_distribution[i]:
                label_distribution[i].append(cls)
            start_idx = end_idx

    # Ensure all data is assigned and there are no out-of-range indices
    for client_id, indices in net_dataidx_map.items():
        out_of_range_indices = [idx for idx in indices if idx < 0 or idx >= len(dataset)]
        if out_of_range_indices:
            print(f"Client {client_id} has out-of-range indices: {out_of_range_indices}")
        assert all(0 <= idx < len(dataset) for idx in indices), f"Client {client_id} has out-of-range indices!"

    # Verify no overlapping indices
    all_indices = [idx for indices in net_dataidx_map.values() for idx in indices]
    if len(all_indices) != len(set(all_indices)):
        print("Overlap detected in data indices!")
        overlapping_indices = set([x for x in all_indices if all_indices.count(x) > 1])
        print(f"Overlapping indices: {overlapping_indices}")
    assert len(all_indices) == len(set(all_indices)), "Overlap detected in data indices!"

    # Print label distribution for each client
    for client_id, labels in label_distribution.items():
        print(f"Client {client_id}: {labels}, Total samples: {local_number_data[client_id]}")

    return net_dataidx_map, local_number_data, label_distribution

