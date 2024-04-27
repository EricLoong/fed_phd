import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import Lambda
from torch.utils.data import Dataset, Subset
from termcolor import colored
import ssl
import os
from glob import glob
from PIL import Image
import json
from utils.data.cifar10 import CIFAR10,partition_data_indices_cifar10

ssl._create_default_https_context = ssl._create_unverified_context


class CustomDataset(Dataset):
    def __init__(self, folder, transform, exts=['jpg', 'jpeg', 'png', 'tiff']):
        super().__init__()
        self.paths = [p for ext in exts for p in glob(os.path.join(folder, f'*.{ext}'))]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img = Image.open(self.paths[item])
        return self.transform(img)


def dataset_wrapper(dataset, data_dir, image_size, augment_horizontal_flip=True, info_color='green', min1to1=True, partial_data=False, net_dataidx_map=None):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip() if augment_horizontal_flip else Lambda(lambda x: x),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if min1to1 else Lambda(lambda x: x)
    ])

    if os.path.isdir(dataset):
        print(colored('Loading local file directory', info_color))
        dataSet = CustomDataset(dataset, transform)
        print(colored('Successfully loaded {} images!'.format(len(dataSet)), info_color))
    else:
        dataset = dataset.lower()
        assert dataset in ['cifar10'], "Dataset must be 'cifar10' or a valid directory path."
        print(colored('Loading {} dataset'.format(dataset), info_color))
        if dataset == 'cifar10':
            train_set = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
            test_set = CIFAR10(root=data_dir, train=False, download=True, transform=transform)
            #fullset = torch.utils.data.ConcatDataset([train_set, test_set])

            if partial_data and type(net_dataidx_map) !=type(None):
                indices = net_dataidx_map
                dataSet = Subset(train_set, indices)
                print(colored(f'Partitioned CIFAR10 Dataset: {len(dataSet)} images.', info_color))
            else:
                dataSet = train_set
                print(colored(f'Loaded CIFAR10 dataset with {len(dataSet)} images.', info_color))
        else:
            raise ValueError('Dataset not supported')

    return dataSet



def save_partition_map(datadir, net_dataidx_map, local_number_data):
    partition_data = {
        'index_map': net_dataidx_map,
        'data_count': local_number_data
    }
    with open(os.path.join(datadir, 'partition_map.json'), 'w') as fp:
        json.dump(partition_data, fp)

def load_partition_map(datadir):
    with open(os.path.join(datadir, 'partition_map.json'), 'r') as fp:
        partition_data = json.load(fp)
    return partition_data['index_map'], partition_data['data_count']
