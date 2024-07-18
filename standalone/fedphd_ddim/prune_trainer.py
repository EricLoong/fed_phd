import time
import os
import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from functools import partial
from tqdm import tqdm
import datetime
from termcolor import colored
from utils.centralized_src.tools import num_to_groups
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def cycle_with_label(dl):
    while True:
        for data in dl:
            img, label = data
            yield img, label

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer:
    def __init__(self, diffusion_model, fid_scorer, inception_scorer,batch_size=32, lr=2e-5, ddim_samplers=None,
                 num_samples=25, result_folder='./results', cpu_percentage=0,
                 ddpm_fid_score_estimate_every=None, ddpm_num_fid_samples=None,
                 max_grad_norm=1., logger=None, args=None, clip=True):
        """
        Trainer for Diffusion model.
        """
        now = datetime.datetime.now()
        self.cur_time = now.strftime('%Y-%m-%d_%Hh%Mm')
        self.logger = logger
        self.args = args
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.nrow = int(math.sqrt(self.num_samples))
        assert (self.nrow ** 2) == self.num_samples, 'num_samples must be a square number. ex) 25, 36, 49, ...'
        self.image_size = self.diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.result_folder = os.path.join(result_folder, self.args.dataset, self.cur_time)
        self.ddpm_result_folder = os.path.join(self.result_folder, 'DDPM')
        self.device = self.diffusion_model.device
        self.clip = clip
        self.ddpm_fid_flag = True if ddpm_fid_score_estimate_every is not None else False
        self.ddpm_is_flag = True if ddpm_fid_score_estimate_every is not None else False
        # IS compuation is followed by FID computation
        self.ddpm_fid_score_estimate_every = ddpm_fid_score_estimate_every
        self.cal_fid = args.calculate_fid
        self.cal_is = args.calculate_is
        self.tqdm_sampler_name = None
        self.tensorboard_name = None
        self.writer = None
        self.global_step = 0
        #self.fid_score_log = dict()
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if clip is True or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'clip'), exist_ok=True)
        if clip is False or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'no_clip'), exist_ok=True)
        os.makedirs(self.result_folder, exist_ok=True)
        self.optimizer = Adam(self.diffusion_model.parameters(), lr=lr)

        if self.args.warmup_steps > 0:
            self.scheduler = LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: min((step + 1) / self.args.warmup_steps, 1.0)
            )
        else:
            self.scheduler = None

        # DDIM sampler setting
        self.ddim_sampling_schedule = list()
        for idx, sampler in enumerate(self.ddim_samplers):
            sampler.sampler_name = 'DDIM_{}_steps{}_eta{}'.format(idx + 1, sampler.ddim_steps, sampler.eta)
            self.ddim_sampling_schedule.append(sampler.sample_every)
            save_path = os.path.join(self.result_folder, sampler.sampler_name)
            sampler.save_path = save_path
            if sampler.save:
                os.makedirs(save_path, exist_ok=True)
            if sampler.generate_image:
                if sampler.clip is True or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'clip'), exist_ok=True)
                if sampler.clip is False or sampler.clip == 'both':
                    os.makedirs(os.path.join(save_path, 'no_clip'), exist_ok=True)
            if sampler.calculate_fid:
                self.cal_fid = True
                if self.tqdm_sampler_name is None:
                    self.tqdm_sampler_name = sampler.sampler_name
                sampler.num_fid_sample = sampler.num_fid_sample if sampler.num_fid_sample is not None else 0
                #self.fid_score_log[sampler.sampler_name] = list()
            if sampler.fixed_noise:
                sampler.register_buffer('noise', torch.randn([self.num_samples, sampler.channel,
                                                              sampler.image_size, sampler.image_size]))

        # Image generation log
        print(colored('Image will be generated with the following sampler(s)', 'cyan'))
        for sampler in self.ddim_samplers:
            if sampler.generate_image:
                print(colored('-> {} / Image generation every {} steps / Fixed Noise : {}'
                              .format(sampler.sampler_name, sampler.sample_every, sampler.fixed_noise), 'cyan'))
        print('\n')

        # FID score
        if not self.cal_fid:
            print(colored('No FID evaluation will be executed!\n'
                          'If you want FID evaluation consider using DDIM sampler.', 'magenta'))
        else:
            self.fid_scorer = fid_scorer

        if not self.cal_is:
            print(colored('No IS evaluation will be executed!\n'
                          'If you want IS evaluation consider using DDIM sampler.', 'magenta'))
        else:
            self.inception_scorer = inception_scorer

    def print_model_param_sum(self, model, msg="Model parameter sum"):
        total_sum = sum(p.sum().item() for p in model.parameters())
        print(f"{msg}: {total_sum}")

    def get_model_params(self):
        model_parameters = self.diffusion_model.state_dict()
        return copy.deepcopy(model_parameters)

    def set_model_params(self, model_parameters):
        self.diffusion_model.load_state_dict(model_parameters)

    def set_data_loader(self, data_loader):
        #self.dataLoader = cycle_with_label(data_loader)  # This procedure is for traditional centralized training. We don't need this for federated learning.
        self.dataLoader = data_loader

    def set_id(self, trainer_id):
        self.id = trainer_id

    def train(self, round_idx):
        epochs = self.args.epochs
        for epoch in range(epochs):
            self.diffusion_model.train()
            epoch_loss = 0  # Initialize variable to accumulate loss
            num_batches = 0  # To count the number of batches

            self.logger.info(f"Starting epoch {epoch + 1}/{epochs}")

            for batch_idx, (image, _) in enumerate(self.dataLoader):  # Iterate through all batches
                self.optimizer.zero_grad()
                image = image.to(self.device)
                loss = self.diffusion_model(image)
                loss.backward()
                nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()  # Step the scheduler after each batch

                epoch_loss += loss.item()  # Accumulate the loss
                num_batches += 1
                #if batch_idx % 10 == 0:
                #    self.logger.info(f"Batch {batch_idx}: Loss {loss.item()}")

            # Calculate the average loss for the epoch
            average_loss = epoch_loss / num_batches

            #if round_idx % self.args.sample_every == 0:
            self.logger.info(f"Round {round_idx} Epoch {epoch} Average Loss: {average_loss}")

            if self.args.central_train:
                self.ddim_image_generation(epoch)
                self.ddim_fid_calculation(epoch)

    def ddim_image_generation(self, current_step):
        with torch.no_grad():
            for sampler in self.ddim_samplers:
                if current_step % sampler.sample_every == 0:
                    print(f"Generating images at step {current_step}")
                    batches = num_to_groups(self.num_samples, self.batch_size)
                    c_batch = np.insert(np.cumsum(np.array(batches)), 0, 0)
                    imgs = []
                    for i, j in zip([True, False], ['clip', 'no_clip']):
                        if sampler.clip not in [i, 'both']:
                            continue
                        for b in range(len(batches)):
                            if sampler.fixed_noise:
                                imgs.append(sampler.sample(self.diffusion_model, batch_size=None, clip=i,
                                                           noise=sampler.noise[c_batch[b]:c_batch[b + 1]]))
                            else:
                                imgs.append(sampler.sample(self.diffusion_model, batch_size=batches[b], clip=i))
                        imgs = torch.cat(imgs, dim=0)
                        save_image(imgs, nrow=self.nrow,
                                   fp=os.path.join(sampler.save_path, j, f'sample_step_{current_step}.png'))
                    self.logger.info(f"Images generated using {sampler.sampler_name} saved.")

    def ddim_fid_calculation(self, current_step):
        with torch.no_grad():
            for sampler in self.ddim_samplers:
                if sampler.calculate_fid and (current_step+1) % self.args.fid_freq == 0:
                    print(f"Calculating FID at step {current_step}")
                    sample_func = partial(sampler.sample, self.diffusion_model)
                    ddim_cur_fid, _ = self.fid_scorer.fid_score(sample_func, sampler.num_fid_sample)
                    self.logger.info(f"FID score using {sampler.sampler_name} at step {current_step}: {ddim_cur_fid}")
                    if sampler.best_fid[0] > ddim_cur_fid:
                        sampler.best_fid[0] = ddim_cur_fid
                        if sampler.save:
                            self.save_model(current_step, sampler.sampler_name, ddim_cur_fid)

    def ddim_inception_calculation(self, current_step):
        with torch.no_grad():
            for sampler in self.ddim_samplers:
                if sampler.calculate_inception and (current_step + 1) % self.args.fid_freq == 0:
                    # Calculate the Inception Score at the same time of FID calculation
                    print(f"Calculating Inception Score at step {current_step}")
                    sample_func = partial(sampler.sample, self.diffusion_model)
                    ddim_cur_inception_mean, ddim_cur_inception_std = self.inception_scorer.inception_score(sample_func, sampler.num_inception_sample)
                    self.logger.info(f"Inception Score using {sampler.sampler_name} at step {current_step}: Mean {ddim_cur_inception_mean}, Std {ddim_cur_inception_std}")
                    # Uncomment the following lines if you want to save the model based on Inception Score
                    # if sampler.best_inception[0] < ddim_cur_inception_mean:
                    #     sampler.best_inception[0] = ddim_cur_inception_mean
                    #     if sampler.save:
                    #         self.save_model(current_step, sampler.sampler_name, ddim_cur_inception_mean)

    def save_model(self, step, sampler_name, fid_score):
        # Construct a filename that includes the step, sampler name, and FID score
        model_filename = f"{self.result_folder}/model_{sampler_name}_step_{step}_fid_{fid_score:.4f}.pt"
        # Save the model's state dictionary
        torch.save(self.diffusion_model.state_dict(), model_filename)
        self.logger.info(f"Model saved at step {step} with FID {fid_score:.4f} using {sampler_name}")
