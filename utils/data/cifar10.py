

import numpy as np
from torchvision.datasets import CIFAR10

def partition_data_indices_cifar10(datadir, partition, n_nets, n_cls):
    # Load CIFAR-10 dataset labels to determine partitions
    cifar10_train_ds = CIFAR10(datadir, train=True, download=True)
    y_train = cifar10_train_ds.targets

    # Initialize the data index map
    net_dataidx_map = {}
    local_number_data = {}

    # Gather indices for each class
    class_indices = [np.where(np.array(y_train) == i)[0] for i in range(10)]  # Assuming 10 classes

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
        clients_per_class = n_nets * n_cls // 10
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
        # Shuffle data indices for each client to mix classes
        for client_id in net_dataidx_map:
            np.random.shuffle(net_dataidx_map[client_id])

    return net_dataidx_map, local_number_data