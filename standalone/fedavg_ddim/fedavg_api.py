import copy
import logging
import os
import pickle
import random
import pdb
import numpy as np
import torch
from standalone.fedavg_ddim.init_trainer import Trainer
import torchvision
from standalone.fedavg_ddim.client import client
from pathlib import Path
import math

class fedavg_api(object):
    def __init__(self, dataset_info, device, args, model_trainer, logger):
        self.logger = logger
        self.device = device
        self.args = args
        data_map_idx, train_data_local_num_dict  = dataset_info
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict,model_trainer, data_map_idx)
        self.init_stat_info()
        self.results_folder = Path('./results')
        self.best_fid = float('inf')

    def _setup_clients(self, train_data_local_num_dict,model_trainer, data_map_idx):
        self.logger.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_in_total):
            local_trainer = self._model_trainer_locallize(model_trainer=model_trainer,args=self.args,logger=self.logger,partition_index=data_map_idx[client_idx])
            c = client(client_idx,
                       train_data_local_num_dict[client_idx], self.args, self.device, local_trainer, self.logger)
            self.client_list.append(c)
        self.logger.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        w_per_mdls = []
        # Initialization
        for clnt in range(self.args.client_num_in_total):
            w_per_mdls.append(copy.deepcopy(w_global))
        if not self.args.tqdm:
            comm_round_iterable = range(self.args.comm_round)
        else:
            from tqdm import tqdm
            comm_round_iterable = tqdm(range(self.args.comm_round), desc="Comm. Rounds", ncols=100)

        for round_idx in comm_round_iterable:
            self.logger.info("################Communication round : {}".format(round_idx))
            w_locals = []
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            client_indexes = np.sort(client_indexes)

            self.logger.info("client_indexes = " + str(client_indexes))

            for cur_clnt in client_indexes:
                self.logger.info('@@@@@@@@@@@@@@@@ Training Client CM({}): {}'.format(round_idx, cur_clnt))
                # update dataset
                client = self.client_list[cur_clnt]
                # update meta components in personal network
                w_per = client.train(copy.deepcopy(w_global), round_idx)  # Get both model and EMA parameters
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_per)))
                #w_per_mdls[cur_clnt] = copy.deepcopy(w_per)
                # self.logger.info("local weights = " + str(w))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w_per)))

            # update global meta weights
            w_global = self._aggregate(w_locals)
            self.global_evaluation(w_global, round_idx=round_idx)
            torch.cuda.empty_cache()
            #print(torch.cuda.memory_summary())
        return w_global

    def global_evaluation(self, w_global, round_idx):
        # Load global model weights for testing and inference
        self.model_trainer.set_model_params(copy.deepcopy(w_global))
        self.model_trainer.ddim_fid_calculation(round_idx)
        self.model_trainer.ddim_image_generation(round_idx)

    def _check_sampler(self,before=False):
        model_updated = self.model_trainer.get_model_params()
        sampler_params = self.model_trainer.fid_scorer_no_ema.sampler.state_dict()

        total_difference = 0.0
        for param_name in model_updated.keys():
            if param_name in sampler_params:
                # Calculate the sum of absolute differences
                diff = torch.sum(torch.abs(model_updated[param_name] - sampler_params[param_name]))
                total_difference += diff.item()
                #print(f"Difference in {param_name}: {diff.item()}")
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
            # Assuming 'sample' returns a tensor of shape (batch_size, channels, height, width)
            samples = trainer.ema.ema_model.sample(batch_size=16)

            # Combining all sampled images into a single image grid
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
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, _) = w_locals[idx]
            training_num += sample_num
        w_global ={}
        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    w_global[k] = local_model_params[k] * w
                else:
                    w_global[k] += local_model_params[k] * w
        return w_global

    def _aggregate_ema(self, ema_locals):
        training_num = sum(num_samples for num_samples, _ in ema_locals)
        ema_global = {}
        for sample_num, ema_params in ema_locals:
            for k in ema_params.keys():
                if k not in ema_global:
                    ema_global[k] = ema_params[k] * (sample_num / training_num)
                else:
                    ema_global[k] += ema_params[k] * (sample_num / training_num)
        return ema_global

    def _model_trainer_locallize(self,model_trainer, args, logger, partition_index):
        model = copy.deepcopy(model_trainer.diffusion_model)
        ddim_samplers = copy.deepcopy(model_trainer.ddim_samplers)
        local_trainer = Trainer(diffusion_model=model, args=args, logger=logger,
                                ddim_samplers=ddim_samplers, subset_data=True,data_indices=partition_index)
        return local_trainer



    def init_stat_info(self):
        self.stat_info = {}
        self.stat_info["sum_comm_params"] = 0
        self.stat_info["sum_training_flops"] = 0
        self.stat_info["global_fid"] = []
        self.stat_info["final_masks"] = []
