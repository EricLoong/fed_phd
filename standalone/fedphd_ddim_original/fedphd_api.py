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

class fedphd_api(object):
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
        self.edge_servers = self._setup_edge_servers()
        self.edge_models = [(1, copy.deepcopy(self.model_trainer.get_model_params())) for _ in range(self.args.num_edge_servers)]

    def _setup_clients(self, train_data_local_num_dict, data_map_idx):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            # Each client uses the shared model_trainer
            c = Client(client_idx, train_data_local_num_dict[client_idx], self.args, self.device, self.model_trainer, self.logger, data_indices=data_map_idx[client_idx])
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def _setup_edge_servers(self):
        edge_servers = [[] for _ in range(self.args.num_edge_servers)]
        return edge_servers

    def train(self):
        w_global = self.model_trainer.get_model_params()
        if not self.args.tqdm:
            comm_round_iterable = range(self.args.comm_round)
        else:
            from tqdm import tqdm
            comm_round_iterable = tqdm(range(self.args.comm_round), desc="Comm. Rounds", ncols=100)

        for round_idx in comm_round_iterable:
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = [[] for _ in range(self.args.num_edge_servers)]
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total, self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            self.logger.info("client_indexes = " + str(client_indexes))

            # Distribute clients to edge servers
            num_clients_per_edge_server = len(client_indexes) // self.args.num_edge_servers
            edge_server_clients = [[] for _ in range(self.args.num_edge_servers)]
            for i, cur_clnt in enumerate(client_indexes):
                edge_server_idx = i % self.args.num_edge_servers
                edge_server_clients[edge_server_idx].append(cur_clnt)

            for edge_server_idx, clients in enumerate(edge_server_clients):
                for cur_clnt in clients:
                    client = self.client_list[cur_clnt]
                    self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}) on Edge Server {}: {}'.format(round_idx, edge_server_idx, cur_clnt))
                    # Train client based on edge server model
                    w_per = client.train(copy.deepcopy(self.edge_models[edge_server_idx][1]), round_idx)
                    w_locals[edge_server_idx].append((client.get_sample_number(), copy.deepcopy(w_per)))

            # Edge server aggregation
            for edge_idx, edge_server in enumerate(w_locals):
                edge_sever_num_samples_temp = sum([w[0] for w in edge_server])
                if len(edge_server) > 1:
                    self.edge_models[edge_idx] = (edge_sever_num_samples_temp,self._aggregate(edge_server))
                else:
                    self.logger.warning(f"Edge server {edge_idx} received insufficient client updates.")

            # Central server aggregation every 5 rounds
            if (round_idx + 1) % self.args.aggr_freq == 0:
                self.logger.info("########## Aggregating at central server ##########")
                w_global = self._aggregate(self.edge_models)
                self.global_evaluation(w_global, round_idx)
                torch.cuda.empty_cache()

        return w_global

    def global_evaluation(self, w_global, round_idx):
        # Load global model weights for testing and inference
        self.model_trainer.set_model_params(copy.deepcopy(w_global))
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

    def _aggregate(self, w_locals):
        training_num = sum(sample_num for sample_num, _ in w_locals)
        w_global = {}
        for k in w_locals[0][1].keys():
            w_global[k] = sum(local_model_params[k] * (sample_num / training_num) for sample_num, local_model_params in w_locals)
        return w_global

    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["global_fid"] = []
        self.stat_info["final_masks"] = []
