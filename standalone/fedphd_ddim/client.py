import torch
from torch.utils.data import DataLoader
from utils.data.dataset import dataset_wrapper
from multiprocessing import cpu_count
import math
import numpy as np

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
        self.dataSet = dataset_wrapper(self.args.dataset, data_dir=self.args.data_dir,
                                       image_size=self.model_trainer.image_size, partial_data=True,
                                       net_dataidx_map=self.data_indices)
        self.dataLoader = DataLoader(self.dataSet, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        self.label_distribution = self._calculate_label_distribution()

    def _calculate_label_distribution(self):
        label_counts = {label: 0 for label in range(self.num_classes)}
        total_samples = len(self.dataSet)
        for _, label in self.dataSet:
            label_counts[label] += 1
        return {label: count / total_samples for label, count in label_counts.items()}

    def train(self, w_global, round_idx):
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.set_data_loader(self.dataLoader)  # Set the client's DataLoader in the Trainer
        self.model_trainer.train(round_idx)
        w_local = self.model_trainer.get_model_params()
        return w_local

    def get_sample_number(self):
        return self.train_data_local_num

    def calculate_homogeneity_score(self, merged_distribution, target_distribution):
        difference = sum(abs(target_distribution[y] - merged_distribution[y]) ** 2 for y in merged_distribution.keys())
        return 2 - math.sqrt(difference)

    def select_best_edge_server(self, edge_distributions, current_samples, target_distribution):
        score_list = []
        merged_distributions = []
        for i in range(len(edge_distributions)):
            merged_distribution = self._merge_edge_distribution(edge_distributions[i], current_samples=current_samples[i])
            merged_distributions.append(merged_distribution)
        for edge_idx, merged_distribution in enumerate(merged_distributions):
            score = self.calculate_homogeneity_score(merged_distribution, target_distribution)
            score_list.append(score)

        # Calculate probabilities
        b = 0
        scores = np.array(score_list)
        current_samples = np.array(current_samples)
        relu = np.maximum(scores * 10000 - current_samples+b, 0)
        #self.logger.info(f"Client {self.client_idx} scores: {scores}")
        #self.logger.info(f"Client {self.client_idx} current samples: {current_samples}")
        # Add epsilon to ensure non-zero probabilities and normalize
        epsilon = 1e-8
        relu += epsilon
        probabilities = relu / np.sum(relu)

        #self.logger.info(f"Client {self.client_idx} probabilities: {probabilities}")

        # Select edge server based on probabilities
        best_edge_server_idx = np.random.choice(len(edge_distributions), p=probabilities)
        #self.logger.info(f"Client {self.client_idx} selects edge server {best_edge_server_idx} with probability {probabilities[best_edge_server_idx]}")
        return best_edge_server_idx

    def _merge_edge_distribution(self, edge_distribution, current_samples):
        # Merge the edge distribution with the client distribution
        # This function is the same as the one in FedPHD, but now we merge it for future selection rather than the distribution
        # is really merged.
        new_samples = self.get_sample_number()
        client_distribution = self.label_distribution
        total_samples = current_samples + new_samples
        merged_distribution = {}
        for label, prob in client_distribution.items():
            merged_distribution[label] = (edge_distribution[label] * current_samples + prob * new_samples) / total_samples
            # edge_distribution[label] * current_samples is the sum of samples of this label before adding the client
            # prob * new_samples is the number of samples of this label in the new client
        return merged_distribution


