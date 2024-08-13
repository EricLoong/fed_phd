import copy
import logging
import os
import pickle
import random
import numpy as np
import torch
from standalone.fedphd_ddim.prune_trainer import Trainer
import torchvision
from standalone.fedphd_ddim.client import Client
from pathlib import Path
import math


class fedphd_api:
    def __init__(self, dataset_info, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        data_map_idx, train_data_local_num_dict, _ = dataset_info
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.model_trainer = model_trainer
        self.num_classes = self._get_num_classes(self.args.dataset)
        self.server_distribution = self._init_uniform_distribution()
        self.previous_server_distribution = self.server_distribution.copy()
        self._setup_clients(train_data_local_num_dict, data_map_idx)
        self.init_stat_info()
        self.results_folder = Path('./results')
        self.best_fid = float('inf')
        self.edge_servers = self._setup_edge_servers()
        self.edge_models = [(1, copy.deepcopy(self.model_trainer.get_model_params())) for _ in
                            range(self.args.num_edge_servers)]

    def _get_num_classes(self, dataset_name):
        dataset_classes = {
            'cifar10': 10,
            'celeba': 4,
            # Add other datasets here as needed
        }
        return dataset_classes.get(dataset_name.lower(), 10)  # Default to 10 if not found

    def _init_uniform_distribution(self):
        return {y: 1 / self.num_classes for y in range(self.num_classes)}

    def _setup_clients(self, train_data_local_num_dict, data_map_idx):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            # Each client uses the shared model_trainer
            c = Client(client_idx, train_data_local_num_dict[client_idx], self.args, self.device, self.model_trainer,
                       self.logger, data_indices=data_map_idx[client_idx], num_classes=self.num_classes)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def _setup_edge_servers(self):
        edge_servers = [[] for _ in range(self.args.num_edge_servers)]
        return edge_servers

    def _init_edge_server_distribution(self):
        edge_distribution = []
        for i in range(self.args.num_edge_servers):
            edge_distribution.append(self.server_distribution.copy())
        return edge_distribution

    def train(self):
        w_global = self.model_trainer.get_model_params()
        if not self.args.tqdm:
            comm_round_iterable = range(self.args.comm_round)
        else:
            from tqdm import tqdm
            comm_round_iterable = tqdm(range(self.args.comm_round), desc="Comm. Rounds", ncols=100)

        # Initialize edge server distribution
        edge_server_distributions = self._init_edge_server_distribution()
        edge_samples = [0 for _ in range(self.args.num_edge_servers)]

        for round_idx in comm_round_iterable:
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = [[] for _ in range(self.args.num_edge_servers)]
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            self.logger.info("client_indexes = " + str(client_indexes))

            # Set the edge server set for each round
            edge_server_clients = [[] for _ in range(self.args.num_edge_servers)]

            # Reset the edge server distribution after central server aggregation

            # Client select the edge server according to the statistic homogeneity score
            for client_idx in client_indexes:
                client = self.client_list[client_idx]
                edge_server_idx = client.select_best_edge_server(
                    edge_distributions=copy.deepcopy(edge_server_distributions),
                    current_samples=edge_samples, target_distribution=self.server_distribution)
                edge_server_clients[edge_server_idx].append(client_idx)
                self.logger.info('Client {} is assigned to Edge Server {}'.format(client_idx, edge_server_idx))

            temp_edge_models = copy.deepcopy(self.edge_models)
            for edge_server_idx, clients in enumerate(edge_server_clients):
                for cur_clnt in clients:
                    client = self.client_list[cur_clnt]
                    self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}) on Edge Server {}: {}'.format(round_idx,
                                                                                                            edge_server_idx,
                                                                                                            cur_clnt))
                    # Train client based on edge server model
                    print(f"Memory usage before round {round_idx}:")
                    print(torch.cuda.memory_summary())
                    w_per = client.train(copy.deepcopy(temp_edge_models[edge_server_idx][1]), round_idx)
                    print(f"Memory usage after train {round_idx}:")
                    print(torch.cuda.memory_summary())
                    w_locals[edge_server_idx].append((client.get_sample_number(), copy.deepcopy(w_per)))

                    # Update client distribution on the selected edge server
                    edge_server_distributions[edge_server_idx] = self._update_edge_distribution(
                        edge_distribution=copy.deepcopy(edge_server_distributions[edge_server_idx]),
                        client_distribution=client.label_distribution,
                        current_samples=edge_samples[edge_server_idx], new_samples=client.get_sample_number()
                    )
                    edge_samples[edge_server_idx] += client.get_sample_number()

            # Edge server aggregation
            for edge_idx, edge_server in enumerate(w_locals):
                edge_sever_num_samples_temp = sum([w[0] for w in edge_server])
                self.logger.info('Edge Server {} has attached {} samples at round {}'.format(edge_idx, edge_sever_num_samples_temp,round_idx))
                # Avoid empty edge server
                if edge_sever_num_samples_temp > 0:
                    self.edge_models[edge_idx] = (edge_sever_num_samples_temp, self._aggregate(edge_server))

            # Central server aggregation every 5 rounds
            if (round_idx+1)  % self.args.aggr_freq == 0:
                self.logger.info("########## Aggregating at central server ##########")
                w_global = self._aggregate_server(w_locals=self.edge_models, target_distribution=self.server_distribution,
                                                  edge_distributions=edge_server_distributions)
                self.global_evaluation(w_global, round_idx)

                # Reset edge server distribution
                edge_server_distributions = self._init_edge_server_distribution()
                edge_samples = [0 for _ in range(self.args.num_edge_servers)]
                # Update edge models with the global model
                for i in range(self.args.num_edge_servers):
                    self.edge_models[i] = (self.edge_models[i][0], copy.deepcopy(w_global))
                torch.cuda.empty_cache()

        return w_global

    def global_evaluation(self, w_global, round_idx):
        # Load global model weights for testing and inference
        self.model_trainer.set_model_params(copy.deepcopy(w_global))
        self.model_trainer.ddim_inception_calculation(round_idx)
        self.model_trainer.ddim_fid_calculation(round_idx)
        self.model_trainer.ddim_image_generation(round_idx)

    def _check_sampler(self, before=False):
        model_updated = self.model_trainer.get_model_params()
        sampler_params = self.model_trainer.fid_scorer_no_ema.sampler.state_dict()

        total_difference = 0.0
        for param_name in model_updated.keys():
            if param_name in sampler_params:
                # Calculate the sum of absolute differences
                diff = torch.sum(torch.abs(model_updated[param_name] - sampler_params[param_name]))
                total_difference += diff.item()
        if before:
            print(f"Total parameter difference before update: {total_difference}")
        else:
            print(f"Total parameter difference after update: {total_difference}")
        return total_difference

    def save_model_checkpoint(self, w_global, round_idx):
        save_path = self.results_folder / f'global_model_{round_idx}.pt'
        torch.save(w_global, save_path)
        print(f"Saved global model checkpoint at: {save_path}")

    def generate_and_save_samples(self, trainer, round_idx):
        trainer.ema.ema_model.eval()  # Ensure the model is in evaluation mode for sampling
        with torch.inference_mode():
            samples = trainer.ema.ema_model.sample(batch_size=16)
            nrow = 4  # Number of images per row and column
            save_path = self.results_folder / f'samples_{round_idx}.png'
            torchvision.utils.save_image(samples, save_path, nrow=nrow)
            print(f"Saved sample images at: {save_path}")

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _update_edge_distribution(self, edge_distribution, client_distribution, current_samples, new_samples):
        total_samples = current_samples + new_samples
        for label, prob in client_distribution.items():
            edge_distribution[label] = (edge_distribution[label] * current_samples + prob * new_samples) / total_samples
            # edge_distribution[label] * current_samples is the sum of samples of this label before adding the client
            # prob * new_samples is the number of samples of this label in the new client
        return edge_distribution

    def _aggregate(self, w_locals):
        training_num = sum(sample_num for sample_num, _ in w_locals)
        if training_num == 0:
            return {}
        w_global = {}
        for k in w_locals[0][1].keys():
            w_global[k] = sum(
                local_model_params[k] * (sample_num / training_num) for sample_num, local_model_params in w_locals)
        return w_global

    def _aggregate_server(self, w_locals, edge_distributions, target_distribution):
        a = self.args.balance_agg_a
        b = self.args.balance_agg_b
        if not w_locals:  # Check if the list is empty
            return {}
        training_num = sum(sample_num for sample_num, _ in w_locals)
        if training_num == 0:
            return {}
        # Calculate homogeneity scores for each edge distribution
        scores_homo = []
        for edge_distribution in edge_distributions:
            score_homo = self.calculate_homogeneity_score(edge_distribution, target_distribution)
            scores_homo.append(score_homo)

        # Compute weights using the ReLU function
        weights = []
        for i, (num_samples, _) in enumerate(w_locals):
            relu_value = max(num_samples + a * scores_homo[i] + b, 0)
            weights.append(relu_value)

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Aggregate the global model parameters using the normalized weights
        w_global = {}
        for k in w_locals[0][1].keys():
            w_global[k] = sum(normalized_weights[i] * w_locals[i][1][k] for i in range(len(w_locals)))

        return w_global

    def calculate_homogeneity_score(self, merged_distribution, target_distribution):
        difference = sum(abs(target_distribution[y] - merged_distribution[y]) ** 2 for y in merged_distribution.keys())
        return 2 - math.sqrt(difference)

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["global_fid"] = []
        self.stat_info["final_masks"] = []
