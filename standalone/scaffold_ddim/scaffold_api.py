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
        self._setup_clients(train_data_local_num_dict, data_map_idx)
        self.init_stat_info()
        self.results_folder = Path('./results')
        self.best_fid = float('inf')

    def _setup_clients(self, train_data_local_num_dict, data_map_idx):
        self.logger.info("############ Setup Clients (START) #############")
        # Initialize global control variate with cloned model parameters
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
        self.logger.info("############ Setup Clients (END) #############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        global_control_variate = self.model_trainer.global_control_variate

        comm_round_iterable = (
            range(self.args.comm_round)
            if not self.args.tqdm
            else tqdm(range(self.args.comm_round), desc="Comm. Rounds", ncols=100)
        )

        for round_idx in comm_round_iterable:
            self.logger.info(f"################ Communication Round : {round_idx}")
            delta_w_locals = []
            local_control_variates = []

            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total, self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)
            self.logger.info("client_indexes = " + str(client_indexes))

            for cur_clnt in client_indexes:
                self.logger.info(f'@@@@@@@@@@@@@@@@ Training Client CM({round_idx}): {cur_clnt}')
                client = self.client_list[cur_clnt]

                # Train client and collect delta updates
                delta_w_per, local_control_variate = client.train(
                    copy.deepcopy(w_global),
                    copy.deepcopy(global_control_variate),
                    round_idx
                )
                delta_w_locals.append((client.get_sample_number(), copy.deepcopy(delta_w_per)))
                local_control_variates.append((client.get_sample_number(), local_control_variate))
                del delta_w_per
                torch.cuda.empty_cache()

            # Aggregate model deltas and control variates
            w_global = self._apply_global_update(w_global, delta_w_locals)
            global_control_variate = self._aggregate_control_variates(local_control_variates, len(client_indexes))

            self.global_evaluation(w_global, round_idx=round_idx)
            torch.cuda.empty_cache()
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
        training_num = sum(sample_num for sample_num, _ in delta_w_locals)

        # Initialize delta to apply to global model
        delta_w_global = {k: torch.zeros_like(v) for k, v in w_global.items()}

        # Aggregate deltas
        for sample_num, delta_params in delta_w_locals:
            weight = sample_num / training_num
            for k in delta_params.keys():
                delta_w_global[k] += delta_params[k] * weight

        # Apply the aggregated delta to the global model
        for k in w_global.keys():
            w_global[k] += delta_w_global[k]

        return w_global

    def _aggregate_control_variates(self, local_control_variates, sampled_client_count):
        training_num = sum(num_samples for num_samples, _ in local_control_variates)
        global_control_variate = {}

        for sample_num, control_variate in local_control_variates:
            weight = sample_num / training_num
            for k, v in control_variate.items():
                global_control_variate[k] = global_control_variate.get(k, 0) + v * weight

        # Scale by S/N to adjust the aggregated control variates
        scaling_factor = sampled_client_count / self.args.client_num_in_total
        for k in global_control_variate:
            global_control_variate[k] *= scaling_factor

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
