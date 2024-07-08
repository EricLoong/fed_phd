import torch
from torch.utils.data import DataLoader
from utils.data.dataset import dataset_wrapper
from multiprocessing import cpu_count

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Client:
    def __init__(self, client_idx, train_data_local_num, args, device, model_trainer, logger, data_indices, num_classes):
        self.client_idx = client_idx
        self.train_data_local_num = train_data_local_num
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.logger = logger
        self.data_indices = data_indices
        self.num_classes = num_classes

        # Initialize the dataset and data loader for the client
        self.dataSet = dataset_wrapper(self.args.dataset, data_dir=self.args.data_dir, image_size=self.model_trainer.image_size, partial_data=True, net_dataidx_map=self.data_indices)
        self.dataLoader = DataLoader(self.dataSet, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

    def train(self, w_global, round_idx):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_data_loader(self.dataLoader)  # Set the client's DataLoader in the Trainer
        #print(f"Client {self.client_idx} DataLoader Output Type: {type(next(iter(self.dataLoader)))}")
        self.model_trainer.train(round_idx)
        w_local = self.model_trainer.get_model_params()
        return w_local

    def _calculate_label_distribution(self):
        label_counts = {label: 0 for label in range(self.num_classes)}
        total_samples = len(self.dataSet)
        for _, label in self.dataSet:
            label_counts[label] += 1
        return {label: count / total_samples for label, count in label_counts.items()}

    def get_sample_number(self):
        return self.train_data_local_num
