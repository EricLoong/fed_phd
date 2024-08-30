import os
import sys
import argparse
import logging
import numpy as np
import torch
import random

def set_directory_to_fed_diff():
    pwd = os.getcwd()
    parts = pwd.split(os.path.sep)

    if 'fed_diff' in parts:
        index = parts.index('fed_diff')
        new_path = os.path.sep.join(parts[:index + 1])
        sys.path.insert(0, new_path)
        print(f"Directory set to: {new_path}")
        return new_path
    else:
        print("The directory 'fed_diff' was not found in the current path.")
        return None

base_path = set_directory_to_fed_diff()


from utils.centralized_src.model_original import Unet
from utils.centralized_src.diffusion import GaussianDiffusion, DDIM_Sampler
from utils.centralized_src.diffusers_unet import unet_cifar10_standard, unet_celeba_standard
from utils.centralized_src.tools import Config,setup_fid_scorer,setup_inception_scorer
from utils.data.cifar10 import partition_data_indices_cifar10
from utils.data.celeba import partition_data_indices_celeba
from standalone.fedavg_ddim.fedavg_api import fedavg_api
from standalone.fedavg_ddim.init_trainer import Trainer
from datetime import datetime


def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w',encoding='UTF-8')
    handler.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    #parser.add_argument('--model_name', type=str, default='simple-u', metavar='N',
    #                    help="network architecture, supporting 'simple-u', 'medium-u', 'ddpm-u'")

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celeba'], metavar='N',
                        help='dataset used for training (options: cifar10, celeba)')

    parser.add_argument('--data_dir', type=str, default=os.path.join(base_path, 'data') if base_path else '/nfs/fed_diff/data/', help='Data directory')
    parser.add_argument('--results_dir', type=str, default='./results', help='Results directory')

    parser.add_argument('--partition_method', type=str, default='iid', metavar='N',
                        help="current supporting two types of data partition, one called 'iid' short for identically and independently distributed"
                             "one called 'n_cls' short for how many classes allocated for each client"
                             )
    parser.add_argument(
        "--partition_alpha",
        type=float,
        default=2,
        metavar="PA",
        help="available parameters for data partition method",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        metavar="N",
        help="local batch size for training",
    )
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train')
    parser.add_argument('--client_num_in_total', type=int, default=100, metavar='N',
                        help='number of clients')
    parser.add_argument('--comm_round', type=int, default=500000, metavar='N',
                        help='number of communication round')
    parser.add_argument('--frac', type=float, default=0.1, metavar='N',
                        help='ratio of clients to join in each round')
    parser.add_argument(
        "--client_optimizer", type=str, default="Adam", help="adam"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="learning rate (default: 0.00001)",
    )

    parser.add_argument('--sampling_steps', type=int, default=100, help='Number of steps to sample from the diffusion model')
    parser.add_argument('--sample_every', type=int, default=500, help='Sample every n steps')
    parser.add_argument('--calculate_fid', action='store_true', help='Calculate FID during training')
    parser.add_argument('--num_fid_sample', type=int, default=30000, help='Number of samples to use for FID calculation')
    parser.add_argument('--calculate_is', action='store_true', help='Calculate Inception Score during training')
    parser.add_argument('--save', action='store_true', help='Save samples during training')
    parser.add_argument('--num_samples', type=int, default=36, help='Number of samples to generate during training')
    parser.add_argument('--cpu_percentage', type=float, default=0.0,
                        help='Percentage of CPU cores to use for data loading')
    parser.add_argument('--fid_estimate_batch_size', type=int, default=256, help='Batch size for FID estimation')
    parser.add_argument('--clip', type=bool, default=True, help='Clip images during generation')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument('--warmup_steps', type=int, default=100, help='Number of warmup steps for the scheduler;5000 for cifar10')
    parser.add_argument('--fid_freq', type=int, default=500, help='Frequency of FID calculation')
    parser.add_argument('--central_train', action='store_true', help='Train a centralized model')
    parser.add_argument('--gradient_accumulate_every', type=int, default=1,
                        help='Number of steps for gradient accumulation')

    # FedProx arguments
    parser.add_argument('--prox', action='store_true', help='Train with proximal term')
    parser.add_argument('--mu', type=float, default=0.1, help='Proximal term scale')
    return parser


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_ddim_sampler(args, diffusion_model):
    # Setup for a single DDIM sampler based on command line arguments
    sampler_config = {
        'ddim_sampling_steps': args.sampling_steps,
        'sample_every': args.sample_every,
        'calculate_fid': args.calculate_fid,
        'num_fid_sample': args.num_fid_sample,
        'save': args.save
    }

    # Initialize the sampler and add it to a list
    ddim_sampler = [DDIM_Sampler(diffusion_model, **sampler_config)]
    return ddim_sampler

def load_model(args,out_unet=False):
    # if args.dataset == "cifar10":
    #    image_size = 32
    #    unet_cifar10 = Unet(dim=128,dim_multiply=(1,2,2,2),image_size=image_size,attn_resolutions=(16,),dropout=0.1,num_res_blocks=2)
    #    diffusion = GaussianDiffusion(unet_cifar10, image_size=image_size).to(args.device)
    if args.dataset == "celeba":
        image_size = 64
        unet = unet_celeba_standard.to(args.device)
        #unet.to(args.device)
        # unet_celeba = Unet(dim=128,dim_multiply=(1,2,2,2),image_size=image_size,attn_resolutions=(16,),dropout=0.0,num_res_blocks=2)
        diffusion = GaussianDiffusion(unet, image_size=image_size)
    elif args.dataset == "cifar10":
        image_size = 32
        unet = unet_cifar10_standard.to(args.device)
        #unet.to(args.device)
        diffusion = GaussianDiffusion(unet, image_size=image_size)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    model = diffusion
    if out_unet:
        return unet
    else:
        return model.to(args.device)

def setup_trainer(args, diffusion_model, fid_scorer,inception_scorer, ddim_samplers,logger):
    # Initialize the trainer with the provided arguments
    trainer = Trainer(args=args,logger=logger,
        diffusion_model=diffusion_model,
        fid_scorer=fid_scorer,
        inception_scorer=inception_scorer,
        batch_size=args.batch_size,
        lr=args.lr,
        num_samples=args.num_samples,
        result_folder=args.results_dir,
        cpu_percentage=args.cpu_percentage,
        ddim_samplers=ddim_samplers,
        clip=args.clip
    )
    return trainer


if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = add_args(argparse.ArgumentParser(description="FedAvg-standalone"))
    args = parser.parse_args()
    device = args.device
    # print("torch version{}".format(torch.__version__))

    Config.initialize(args)

    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
    data_partition = args.partition_method
    if data_partition != "iid":
        data_partition += str(args.partition_alpha)
    if args.prox:
        args.identity = "fedprox" + "-" + data_partition
    else:
        args.identity = "fedavg" + "-" + data_partition
    args.identity += "-{}".format(args.dataset)
    args.client_num_per_round = int(args.client_num_in_total * args.frac)
    #args.identity += "-mdl" + args.model_name
    args.identity += (
        "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    )
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += "-batchsize" + str(args.batch_size*args.gradient_accumulate_every)
    args.identity += "-seed" + str(args.seed)

    cur_dir = os.path.abspath(__file__).rsplit("/", 1)[0]
    log_path = os.path.join(
        cur_dir,
        "LOG/"
        + args.dataset
        + "/"
        + current_datetime_str
        + args.identity
        + ".log",
    )
    logger = logger_config(log_path=log_path, logging_name=args.identity)
    logger.info(args)


    # Set the random seed. The np.random seed determines the dataset partition.
    # The torch_manual_seed determines the initial weight.
    # We fix these two, so that we can reproduce the result.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load data and generated image prepared for FID calculation
    print('Dataset dir is :', args.data_dir)
    # create model.
    diffusion_model = load_model(args)
    num_params = num_params(diffusion_model)
    logger.info("Model num of params:{}".format(num_params))
    # print(model)
    # pretrained_model_path = os.path.join(cur_dir, 'results', args.dataset, '20240404_005104fedavg-iid-mdlmedium-u-cm50000-total_clnt1-neighbor1-seed2023.pth')
    ddim_samplers = setup_ddim_sampler(args, diffusion_model) # Just one sampler in defalt
    fid_scorer = setup_fid_scorer(args,image_size=diffusion_model.image_size)
    inception_scorer = setup_inception_scorer(args)
    global_model_trainer = setup_trainer(args, diffusion_model, fid_scorer=fid_scorer,inception_scorer=inception_scorer, ddim_samplers=ddim_samplers,logger=logger)
    logger.info(diffusion_model)
    if args.dataset == "celeba":
        data_info = partition_data_indices_celeba(datadir=args.data_dir, partition=args.partition_method, n_nets=args.client_num_in_total)
    elif args.dataset == "cifar10":
        data_info = partition_data_indices_cifar10(datadir=args.data_dir, partition=args.partition_method, n_nets=args.client_num_in_total, n_cls=args.partition_alpha)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    FedAvgAPI = fedavg_api(data_info, device, args, global_model_trainer, logger)
    final_global_model = FedAvgAPI.train()