import numpy as np
import pandas as pd
import os
from torchvision.datasets import CelebA
from torch.utils.data import Dataset
from torchvision import transforms
import random


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
    # Load CelebA dataset and attributes
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CelebADataset(root=datadir, split='train', transform=transform)
    y_train = dataset.attr[['Male', 'Young']].apply(create_classes, axis=1).values

    # Count samples for each class
    class_counts = np.bincount(y_train, minlength=4)
    total_samples = sum(class_counts)
    class_probabilities = class_counts / total_samples

    for i in range(4):
        print(f"Class {i}: {class_counts[i]} samples")

    # Initialize the data index map and label distribution map
    net_dataidx_map = {i: [] for i in range(n_nets)}
    label_distribution = {client_id: [] for client_id in range(n_nets)}

    # Gather indices for each class
    class_indices = [np.where(y_train == i)[0] for i in range(4)]  # 4 classes based on attributes

    # Assign data to clients
    for i in range(n_nets):
        selected_classes = np.random.choice(4, 2, p=class_probabilities, replace=False)
        for cls in selected_classes:
            cls_indices = class_indices[cls]
            np.random.shuffle(cls_indices)
            num_samples = len(cls_indices) // (n_nets // 2)  # Divide samples equally among clients
            assigned_samples = cls_indices[:num_samples]
            class_indices[cls] = cls_indices[num_samples:]  # Remove assigned samples from class indices
            net_dataidx_map[i].extend(assigned_samples)
            label_distribution[i].append(cls)
            if len(assigned_samples) == 0:
                class_probabilities[cls] = 0
            else:
                class_probabilities[cls] = len(class_indices[cls]) / total_samples

    # Ensure all data is assigned
    remaining_indices = [index for indices in class_indices for index in indices]
    for i in range(n_nets):
        if len(remaining_indices) == 0:
            break
        additional_sample = remaining_indices.pop(0)
        net_dataidx_map[i].append(additional_sample)

    # Print label distribution for each client
    for client_id, labels in label_distribution.items():
        print(f"Client {client_id}: {labels}")

    return net_dataidx_map, label_distribution



