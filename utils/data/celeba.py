import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd

# Create a custom dataset class for CelebA-HQ, which is the usage as CIFAR10.

class CelebaHQ(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.zip_file_path = os.path.join(self.root, 'celeba_images.zip')

        with zipfile.ZipFile(self.zip_file_path, 'r') as z:
            # Assuming images are directly in the root of the zip file
            self.image_ids = [info.filename for info in z.infolist() if info.filename.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img_name = self.image_ids[index]
        with zipfile.ZipFile(self.zip_file_path, 'r') as z:
            with z.open(img_name) as img_file:
                image = Image.open(img_file).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

import numpy as np

def partition_data_indices_celeba(datadir, partition, n_nets, n_cls):
    # Load CelebA-HQ dataset attributes
    celeba_dataset = CelebaHQ(datadir, train=True)
    y_train = [attr[0] for _, attr in celeba_dataset]  # Assuming using the first attribute for simplicity

    # Initialize the data index map, local data number map, and label distribution map
    net_dataidx_map = {}
    local_number_data = {}
    label_distribution = {client_id: [] for client_id in range(n_nets)}

    # Gather indices for each class
    unique_classes = list(set(y_train))
    class_indices = [np.where(np.array(y_train) == i)[0] for i in unique_classes]

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
        clients_per_class = n_nets * n_cls // len(unique_classes)
        for i, cls_idx in enumerate(class_indices):
            np.random.shuffle(cls_idx)
            split_size = len(cls_idx) // clients_per_class
            for j in range(clients_per_class):
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

# Example usage
#if __name__ == '__main__':
#    datadir = '~/data/celeba'
#    partition = 'iid'
#    n_nets = 5
#    n_cls = 2

#    net_dataidx_map, local_number_data, label_distribution = partition_data_indices_celeba(datadir, partition, n_nets, n_cls)
#    print(net_dataidx_map)
#    print(local_number_data)
#    print(label_distribution)



