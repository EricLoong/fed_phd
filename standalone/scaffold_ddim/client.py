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

        # Initialize the dataset and data loader for the client
        self.dataSet = dataset_wrapper(
            self.args.dataset,
            data_dir=self.args.data_dir,
            image_size=self.model_trainer.image_size,
            partial_data=True,
            net_dataidx_map=self.data_indices
        )
        self.dataLoader = DataLoader(self.dataSet, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

    def train(self, w_global, global_control_variate, local_control_variate, round_idx):
        # Set the global model parameters and control variate received from the server
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_data_loader(self.dataLoader)
        self.model_trainer.global_control_variate = global_control_variate

        # Train the model with SCAFFOLD adjustments
        delta_w, delta_c, updated_local_control_variate = self.model_trainer.train(round_idx, local_control_variate)

        # Return the updated model delta, control variate delta, and the new local control variate
        return delta_w, delta_c, updated_local_control_variate

    def get_sample_number(self):
        return self.train_data_local_num
