import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from ignite.metrics import FID, InceptionScore
import numpy as np
from .init_trainer import Trainer
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation
from pathlib import Path
from torch.optim import Adam
from torch.utils.data import TensorDataset,Dataset, DataLoader
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import LambdaLR


class FederatedTrainer(Trainer):
    def __init__(self, diffusion_model, folder, data_partition_map, args=None, logger=None, **kwargs):
        super().__init__(diffusion_model, folder, **kwargs)
        self.args = args
        self.batch_size = self.args.batch_size
        self.logger = logger
        self.model = diffusion_model
        self.calculate_fid = args.test_fid
        self.num_fid_samples = args.num_fid
        self.save_best_and_latest_only = True
        if type(data_partition_map) != type(None):
            dl_local = self.get_client_dataloaders(self.ds, data_partition_map)
            dl_local = self.accelerator.prepare(dl_local)
            self.dl_local = cycle(dl_local)

        adam_betas = (0.9, 0.999)
        self.opt = Adam(diffusion_model.parameters(), lr=self.args.lr, betas=adam_betas)
        if self.args.warmup_steps > 0:
            self.scheduler = LambdaLR(
                self.opt,
                lr_lambda=lambda step: min((step + 1) / self.args.warmup_steps, 1.0)
            )
        else:
            self.scheduler = None

        # redefine the fid_scorer
        self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir='./results',
                device=self.device,
                num_fid_samples=self.args.num_fid,
                inception_block_idx=2048
            )


        # Makre sure didn't change the original trainer, we redefine the fid_scorer

    def update_gm_fid_scorer(self):
        #print("Updating FID scorer with new model parameters...")
        if self.calculate_fid:
            self.fid_scorer_no_ema = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir='./results',
                device=self.device,
                num_fid_samples=self.args.num_fid,
                inception_block_idx=2048
            )
        # Debugging output
        #self.print_model_param_sum(self.model, "New model in FID scorer after update")

    def print_model_param_sum(self, model, msg="Model parameter sum"):
        total_sum = sum(p.sum().item() for p in model.parameters())
        print(f"{msg}: {total_sum}")


    def set_id(self, trainer_id):
        self.id = trainer_id

    def get_model_params(self):
        model_parameters = self.model.state_dict()
        return copy.deepcopy(model_parameters)

    def get_ema_params(self):
        model_parameters = self.ema.model.state_dict()
        return copy.deepcopy(model_parameters)

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_ema_model_params(self, model_parameters):
        self.ema.ema_model.load_state_dict(model_parameters)

    def cycle(self,dl):
        while True:
            for data in dl:
                yield data

    def train_local(self,round_idx=0):
        device = self.device  # Ensure device setup is correct
        accelerator = self.accelerator
        #self.opt = Adam(
        #    filter(lambda p: p.requires_grad, self.model.parameters()),
        #    lr=self.args.lr * (self.args.lr_decay ** round_idx),
        #    betas=adam_betas,
        #    weight_decay=self.args.wd,
        #)

        #self.opt = Adam(self.model.parameters(), lr=self.args.lr, betas=adam_betas)
        #self.opt= self.accelerator.prepare(self.opt)

        # Start the modified training loop, iterating for a fixed number of epochs
        for epoch in range(self.args.epochs):
            total_loss = 0.0
            for step in range(self.args.gradient_accumulate_every):
                # Assuming your DataLoader yields batches in the correct format
                data = next(self.dl_local)

                with self.accelerator.autocast():
                    loss = self.model(data)  #
                    loss = loss / self.args.gradient_accumulate_every
                    total_loss += loss.item()

                # Backward pass and optimization steps, adapted from original 'Trainer'
                self.accelerator.backward(loss)

            accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.opt.step()
            self.opt.zero_grad()
            if self.scheduler:
                self.scheduler.step()  # Step the scheduler after each batch

            self.logger.info(f'Epoch: {epoch}, Step: {step}, Loss: {total_loss:.6f}, LR: {self.scheduler.get_last_lr()[0] if self.scheduler else self.args.lr:.6f}')
            self.ema.update()

        #    if round_idx%self.args.test_interval == 0 and round_idx != 0:
        #        self.ema.ema_model.eval()
        #        fid_score = self.fid_scorer.fid_score()
        #        accelerator.print(f'fid_score: {fid_score}')
        #        self.logger.info(f'fid_score locally: {fid_score}')
        #fid_score = self.fid_scorer.fid_score()
        #self.accelerator.print(f'fid_score: {fid_score}')
        #self.logger.info(f'fid_score: {fid_score}')


    def get_client_dataloaders(self,full_dataset, net_dataidx_map):
        print(net_dataidx_map)
        client_dataset = Subset(full_dataset, net_dataidx_map)
        client_loader = DataLoader(client_dataset, batch_size=self.batch_size, shuffle=True)
        return client_loader

def cycle(dl):
    while True:
        for data in dl:
            yield data

