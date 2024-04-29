# Modified trainer for federated learning from the Trainer in utils/centralized_src/trainer.py
import time
import os
import math
import copy
from utils.data.dataset import dataset_wrapper
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from multiprocessing import cpu_count
from functools import partial
from tqdm import tqdm
import datetime
from termcolor import colored
from utils.centralized_src.tools import FID,num_to_groups
import numpy as np
from torch.optim.lr_scheduler import LambdaLR

def cycle_with_label(dl):
    while True:
        for data in dl:
            img, label = data
            yield img


def cycle(dl):
    while True:
        for data in dl:
            yield data


class Trainer:
    def __init__(self, diffusion_model, batch_size=32, lr=2e-5, ddim_samplers=None,
                num_samples=25, result_folder='./results', cpu_percentage=0,
                 fid_estimate_batch_size=None, ddpm_fid_score_estimate_every=None, ddpm_num_fid_samples=None,
                 max_grad_norm=1., logger=None, subset_data=False, args=None, clip=True, data_indices=None):
        """
        Trainer for Diffusion model.
        :param diffusion_model: GaussianDiffusion model
        :param args: experiment arguments
        :param batch_size: batch size for training. DDPM author used 128 for cifar10 and 64 for 256X256 image
        :param lr: DDPM author used 2e-4 for cifar10 and 2e-5 for 256X256 image
        :param ddim_samplers: List containing DDIM samplers.
        For example if it is set to 1000, then trainer will save models in every 1000 step and save generated images
        based on DDPM sampling schema. If you want to generate image based on DDIM sampling, you have to pass a list
        containing corresponding DDIM sampler.
        :param num_samples: # of generating images, must be square number ex) 25, 36, 49...
        :param result_folder: where model, generated images will be saved
        :param cpu_percentage: The percentage of CPU used for Dataloader i.e. num_workers in Dataloader.
        Value must be [0, 1] where 1 means using all cpu for dataloader. If you are Windows user setting value other
        than 0 will cause problem, so set to 0
        :param fid_estimate_batch_size: batch size for FID calculation. It has nothing to do with training.
        :param ddpm_fid_score_estimate_every: Step interval for FID calculation using DDPM. If set to None, FID score
        will not be calculated with DDPM sampling. If you use DDPM sampling for FID calculation, it can be very
        time consuming, so it is wise to set this value to None, and use DDIM sampler for FID calculation. But anyway
        you can calculate FID score with DDPM sampler if you insist to.
        :param ddpm_num_fid_samples: # of generating images for FID calculation using DDPM sampler. If you set
        ddpm_fid_score_estimate_every to None, i.e. not using DDPM sampler for FID calculation, then this value will
        be just ignored.
        :param max_grad_norm: Restrict the norm of maximum gradient to this value
        :param exp_name: experiment name. If set to None, it will be decided automatically as folder name of dataset.
        :param clip: [True, False, 'both'] you can find detail in p_sample function in diffusion.py file.
        """

        # Metadata & Initialization & Make directory for saving files.
        now = datetime.datetime.now()
        self.cur_time = now.strftime('%Y-%m-%d_%Hh%Mm')
        self.logger = logger
        self.args = args
        self.dataset = self.args.dataset
        self.exp_name = self.dataset
        self.diffusion_model = diffusion_model
        self.ddim_samplers = ddim_samplers
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.nrow = int(math.sqrt(self.num_samples))
        assert (self.nrow ** 2) == self.num_samples, 'num_samples must be a square number. ex) 25, 36, 49, ...'
        self.image_size = self.diffusion_model.image_size
        self.max_grad_norm = max_grad_norm
        self.result_folder = os.path.join(result_folder, self.exp_name, self.cur_time)
        self.ddpm_result_folder = os.path.join(self.result_folder, 'DDPM')
        self.device = self.diffusion_model.device
        self.clip = clip
        self.ddpm_fid_flag = True if ddpm_fid_score_estimate_every is not None else False
        self.ddpm_fid_score_estimate_every = ddpm_fid_score_estimate_every
        self.cal_fid = True if self.ddpm_fid_flag else False
        self.tqdm_sampler_name = None
        self.tensorboard_name = None
        self.writer = None
        self.global_step = 0
        self.fid_score_log = dict()
        assert clip in [True, False, 'both'], "clip must be one of [True, False, 'both']"
        if clip is True or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'clip'), exist_ok=True)
        if clip is False or clip == 'both':
            os.makedirs(os.path.join(self.ddpm_result_folder, 'no_clip'), exist_ok=True)
        os.makedirs(self.result_folder, exist_ok=True)
        dataset = self.dataset
        # Dataset & DataLoader & Optimizer
        dataSet = dataset_wrapper(dataset,data_dir= self.args.data_dir, image_size=self.image_size, partial_data=subset_data,net_dataidx_map=data_indices)
        assert len(dataSet) >= 100, 'you should have at least 100 images in your folder.at least 10k images recommended'
        print(colored('Dataset Length: {}\n'.format(len(dataSet)), 'green'))
        CPU_cnt = cpu_count()
        # TODO: pin_memory?
        num_workers = int(CPU_cnt * cpu_percentage)
        assert num_workers <= CPU_cnt, "cpu_percentage must be [0.0, 1.0]"
        dataLoader = DataLoader(dataSet, batch_size=self.batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True)
        self.dataLoader = cycle(dataLoader) if os.path.isdir(dataset) else cycle_with_label(dataLoader)
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
                sampler.num_fid_sample = sampler.num_fid_sample if sampler.num_fid_sample is not None else len(dataSet)
                self.fid_score_log[sampler.sampler_name] = list()
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
            self.fid_batch_size = fid_estimate_batch_size if fid_estimate_batch_size is not None else self.batch_size
            dataSet_fid = dataset_wrapper(dataset, image_size=self.image_size,data_dir=self.args.data_dir,
                                          augment_horizontal_flip=False, info_color='magenta', min1to1=False, partial_data=False)
            dataLoader_fid = DataLoader(dataSet_fid, batch_size=self.fid_batch_size, num_workers=num_workers)

            self.fid_scorer = FID(self.fid_batch_size, dataLoader_fid, dataset_name=self.exp_name, device=self.device,
                                  no_label=os.path.isdir(dataset))

            print(colored('FID score will be calculated with the following sampler(s)', 'magenta'))
            if self.ddpm_fid_flag:
                self.ddpm_num_fid_samples = ddpm_num_fid_samples if ddpm_num_fid_samples is not None else len(dataSet)
                print(colored('-> DDPM Sampler / FID calculation every {} steps with {} generated samples'
                              .format(self.ddpm_fid_score_estimate_every, self.ddpm_num_fid_samples), 'magenta'))
            for sampler in self.ddim_samplers:
                if sampler.calculate_fid:
                    print(colored('-> {} / FID calculation every {} steps with {} generated samples'
                                  .format(sampler.sampler_name, sampler.sample_every,
                                          sampler.num_fid_sample), 'magenta'))
            print('\n')
            if self.ddpm_fid_flag:
                self.tqdm_sampler_name = 'DDPM'
                self.fid_score_log['DDPM'] = list()
                msg = """
                FID computation witm DDPM sampler requires a lot of generated samples and can therefore be very time 
                consuming.\nTo accelerate sampling, only using DDIM sampling is recommended. To disable DDPM sampling,
                set [ddpm_fid_score_estimate_every] parameter to None while instantiating Trainer.\n
                """
                print(colored(msg, 'red'))
            del dataLoader_fid
            del dataSet_fid

    def print_model_param_sum(self, model, msg="Model parameter sum"):
        total_sum = sum(p.sum().item() for p in model.parameters())
        print(f"{msg}: {total_sum}")

    def get_model_params(self):
        model_parameters = self.diffusion_model.state_dict()
        return copy.deepcopy(model_parameters)

    def set_model_params(self, model_parameters):
        self.diffusion_model.load_state_dict(model_parameters)

    def set_id(self, trainer_id):
        self.id = trainer_id

    def train(self,round_idx):
        epochs = self.args.epochs
        #print("Starting Training for {} epochs".format(epochs))
        for epoch in range(epochs):
            self.diffusion_model.train()
            self.optimizer.zero_grad()
            image = next(self.dataLoader).to(self.device)
            loss = self.diffusion_model(image)
            loss.backward()
            nn.utils.clip_grad_norm_(self.diffusion_model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()  # Step the scheduler after each batch
            if round_idx % self.args.sample_every == 0:
                self.logger.info(f"Round {round_idx} Loss: {loss.item()}")

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
                if sampler.calculate_fid and current_step % self.args.fid_freq == 0 and current_step != 0:
                    print(f"Calculating FID at step {current_step}")
                    sample_func = partial(sampler.sample, self.diffusion_model)
                    ddim_cur_fid, _ = self.fid_scorer.fid_score(sample_func, sampler.num_fid_sample)
                    self.logger.info(f"FID score using {sampler.sampler_name} at step {current_step}: {ddim_cur_fid}")
                    self.save_model(current_step, sampler.sampler_name, ddim_cur_fid)

    def save_model(self, step, sampler_name, fid_score):
        # Construct a filename that includes the step, sampler name, and FID score
        model_filename = f"{self.result_folder}/model_{sampler_name}_step_{step}_fid_{fid_score:.4f}.pt"

        # Save the model's state dictionary
        torch.save(self.diffusion_model.state_dict(), model_filename)
        self.logger.info(f"Model saved at step {step} with FID {fid_score:.4f} using {sampler_name}")



