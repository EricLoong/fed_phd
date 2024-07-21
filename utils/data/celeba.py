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
    # Load CelebA dataset and attributes
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CelebADataset(root=datadir, split='train', transform=transform)
    y_train = dataset.attr[['Male', 'Young']].apply(create_classes, axis=1).values

    # Count samples for each class
    class_counts = np.bincount(y_train, minlength=4)
    for i in range(4):
        print(f"Class {i}: {class_counts[i]} samples")

    # Initialize the data index map, local data number map, and label distribution map
    net_dataidx_map = {}
    local_number_data = {}
    label_distribution = {client_id: [] for client_id in range(n_nets)}

    # Gather indices for each class
    class_indices = [np.where(y_train == i)[0] for i in range(4)]  # 4 classes based on attributes

    if partition == 'iid':
        all_idxs = np.arange(len(y_train))
        np.random.shuffle(all_idxs)
        data_per_client = len(all_idxs) // n_nets
        for i in range(n_nets):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client if i < n_nets - 1 else len(all_idxs)
            net_dataidx_map[i] = all_idxs[start_idx:end_idx]
            local_number_data[i] = len(net_dataidx_map[i])
    elif partition == 'noniid-pathological':
        # Calculate the number of clients per class, assuming each class must be represented in n_cls clients
        clients_per_class = n_nets * n_cls // 4
        for i, cls_idx in enumerate(class_indices):
            np.random.shuffle(cls_idx)
            split_size = len(cls_idx) // clients_per_class
            for j in range(int(clients_per_class)):
                client_id = (i * clients_per_class + j) % n_nets
                if client_id in net_dataidx_map:
                    net_dataidx_map[client_id] = np.concatenate((net_dataidx_map[client_id], cls_idx[j * split_size:(j + 1) * split_size]))
                else:
                    net_dataidx_map[client_id] = cls_idx[j * split_size:(j + 1) * split_size]
                local_number_data[client_id] = len(net_dataidx_map[client_id])
                if i not in label_distribution[client_id]:
                    label_distribution[client_id].append(i)
        # Shuffle data indices for each client to mix classes
        for client_id in net_dataidx_map:
            np.random.shuffle(net_dataidx_map[client_id])

    # Print label distribution for each client
    for client_id, labels in label_distribution.items():
        print(f"Client {client_id}: {labels}")

    return net_dataidx_map, local_number_data, label_distribution

