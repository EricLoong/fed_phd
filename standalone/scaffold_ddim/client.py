import torch
from torch.utils.data import DataLoader
from utils.data.dataset import dataset_wrapper
from multiprocessing import cpu_count


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Client:
    def __init__(self, client_idx, train_data_local_num, args, device, model_trainer, logger, data_indices):
        self.client_idx = client_idx
        self.train_data_local_num = train_data_local_num
        self.args = args
        self.device = device
        self.model_trainer = model_trainer
        self.logger = logger
        self.data_indices = data_indices

        # Initialize local dataset and data loader
        self._setup_local_dataloader()

    def _setup_local_dataloader(self):
        # Assuming you have a method to get the dataset based on indices
        # Replace with your actual data loading logic
        dataset = self._get_dataset(self.data_indices)
        self.dataLoader = torch.utils.data.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)

    def _get_dataset(self, indices):
        # Replace this with your actual dataset fetching logic
        # This is a placeholder
        return torch.utils.data.Subset(self.args.full_dataset, indices)

    def get_sample_number(self):
        return self.train_data_local_num

    def train(self, w_global, global_control_variate, local_control_variate, round_idx):
        # Update local model parameters to the global parameters
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.global_control_variate = global_control_variate

        # Set the local data loader for the trainer
        self.model_trainer.set_data_loader(self.dataLoader)

        # Perform local training and get updates
        delta_w, delta_c, updated_local_control_variate = self.model_trainer.train(round_idx, local_control_variate)

        return delta_w, delta_c, updated_local_control_variate
