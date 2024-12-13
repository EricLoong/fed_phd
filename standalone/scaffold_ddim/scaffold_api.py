import copy
import logging
import os
import pickle
import random
import numpy as np
import torch
from standalone.scaffold_ddim.scfd_trainer import Trainer
import torchvision
from standalone.scaffold_ddim.client import Client
from pathlib import Path
from tqdm import tqdm

class ScaffoldAPI:
    def __init__(self, dataset_info, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        data_map_idx, train_data_local_num_dict, _ = dataset_info
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.model_trainer = model_trainer
        self.client_control_variates = {}  # Dictionary to store each client's control variate
        self._setup_clients(train_data_local_num_dict, data_map_idx)
        self.init_stat_info()
        self.results_folder = Path('./results')
        self.best_fid = float('inf')
        self.reset_interval = 10  # Interval for resetting control variates

    def _setup_clients(self, train_data_local_num_dict, data_map_idx):
        self.logger.info("############ Setup Clients (START) #############")
        # Initialize global control variate as deepcopy-compatible tensors
        self.model_trainer.global_control_variate = {
            k: v.clone().detach() for k, v in self.model_trainer.get_model_params().items()
        }

        for client_idx in range(self.args.client_num_in_total):
            client = Client(
                client_idx,
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                self.model_trainer,
                self.logger,
                data_indices=data_map_idx[client_idx]
            )
            self.client_list.append(client)
            # Initialize each client's control variate with zeros
            self.client_control_variates[client_idx] = {
                k: torch.zeros_like(v) for k, v in self.model_trainer.get_model_params().items()
            }
        self.logger.info("############ Setup Clients (END) #############")

    def train(self):
        w_global, global_control_variate = self.model_trainer.get_model_and_control_params()

        comm_round_iterable = tqdm(range(self.args.comm_round), desc="Comm. Rounds", ncols=100)

        for round_idx in comm_round_iterable:
            self.logger.info(f"################ Communication round : {round_idx}")
            delta_w_locals = []
            delta_c_locals = []  # To collect control variate deltas

            # Periodic reset of control variates every `reset_interval` rounds
            if round_idx % self.reset_interval == 0:
                self.logger.info("Resetting control variates to zero for stabilization.")
                self.client_control_variates = {
                    client_idx: {k: torch.zeros_like(v) for k, v in control_variate.items()}
                    for client_idx, control_variate in self.client_control_variates.items()
                }
                global_control_variate = {k: torch.zeros_like(v) for k, v in global_control_variate.items()}

            # Sample clients for the current round
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total, self.args.client_num_per_round)
            for client_idx in client_indexes:
                client = self.client_list[client_idx]
                local_control_variate = self.client_control_variates[client_idx]

                # Train client and get model delta, control variate delta, and updated local control variate
                delta_w, delta_c, updated_local_control_variate = client.train(w_global, global_control_variate, local_control_variate, round_idx)

                # Log the norms for debugging purposes
                for name, delta in delta_w.items():
                    self.logger.info(f"Round {round_idx}, Client {client_idx}, Norm of delta_w[{name}]: {torch.norm(delta)}")
                for name, delta in delta_c.items():
                    self.logger.info(f"Round {round_idx}, Client {client_idx}, Norm of delta_c[{name}]: {torch.norm(delta)}")

                delta_w_locals.append((client.get_sample_number(), delta_w))
                delta_c_locals.append((client.get_sample_number(), delta_c))

                # Update the client control variate with the new local control variate
                self.client_control_variates[client_idx] = updated_local_control_variate

            # Aggregate model updates and control variates
            w_global = self._apply_global_update(w_global, delta_w_locals)
            global_control_variate = self._aggregate_control_variates(delta_c_locals, len(client_indexes))

            # Update global parameters
            self.model_trainer.set_model_params(w_global)
            self.model_trainer.global_control_variate = global_control_variate

        return w_global

    def global_evaluation(self, w_global, round_idx):
        self.model_trainer.set_model_params(copy.deepcopy(w_global))
        self.model_trainer.ddim_fid_calculation(round_idx)
        self.model_trainer.ddim_image_generation(round_idx)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = list(range(client_num_in_total))
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        self.logger.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _apply_global_update(self, w_global, delta_w_locals):
        # Ensure all tensors are on the same device as w_global
        device = next(iter(w_global.values())).device  # Get the device of w_global tensors

        training_num = sum(sample_num for sample_num, _ in delta_w_locals)

        # Initialize delta to apply to global model, ensuring it's on the correct device
        delta_w_global = {k: torch.zeros_like(v).to(device) for k, v in w_global.items()}

        # Aggregate deltas
        for sample_num, delta_params in delta_w_locals:
            weight = sample_num / training_num
            for k in delta_params.keys():
                delta_w_global[k] += delta_params[k].to(device) * weight  # Ensure delta_params[k] is on the same device

        # Apply the aggregated delta to the global model
        for k in w_global.keys():
            w_global[k] = w_global[k].to(device) + delta_w_global[k]

        return w_global

    def _aggregate_control_variates(self, local_control_variates, sampled_client_count):
        # Initialize with zeros
        global_control_variate = {
            k: torch.zeros_like(v)
            for k, v in self.model_trainer.get_model_params().items()
        }

        # Aggregate weighted control variates
        total_samples = sum(num_samples for num_samples, _ in local_control_variates)
        for sample_num, control_variate in local_control_variates:
            weight = sample_num / total_samples
            for k, v in control_variate.items():
                global_control_variate[k].add_(v * weight)

        # Apply scaling factor
        scaling_factor = sampled_client_count / self.args.client_num_in_total
        for k in global_control_variate:
            global_control_variate[k].mul_(scaling_factor)

        return global_control_variate

    def save_model_checkpoint(self, w_global, round_idx):
        save_path = self.results_folder / f'global_model_{round_idx}.pt'
        torch.save(w_global, save_path)
        print(f"Saved global model checkpoint at: {save_path}")

    def init_stat_info(self):
        self.stat_info = {
            "sum_comm_params": 0,
            "sum_training_flops": 0,
            "global_fid": [],
            "final_masks": []
        }
