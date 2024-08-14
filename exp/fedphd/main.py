import os
import sys
import argparse
import logging
import numpy as np
import torch
import copy
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

# Set the directory to the root of the project
# This is to ensure that the project modules can be imported
base_path = set_directory_to_fed_diff()

# from utils.centralized_src.model_original import Unet
from utils.centralized_src.diffusion import GaussianDiffusion, DDIM_Sampler
from utils.centralized_src.diffusers_unet import unet_cifar10_standard, unet_celeba_standard
from utils.centralized_src.tools import Config, setup_fid_scorer, setup_inception_scorer
from utils.data.cifar10 import partition_data_indices_cifar10
from utils.data.celeba import partition_data_indices_celeba
from standalone.fedphd_ddim.fedphd_api import fedphd_api
from standalone.fedphd_ddim.prune_trainer import Trainer
from datetime import datetime
from standalone.fedphd_ddim.structure_prune import group_norm_prune
def logger_config(log_path, logging_name):
    logger = logging.getLogger(logging_name)
    logger.setLevel(level=logging.DEBUG)
    handler = logging.FileHandler(log_path, mode='w', encoding='UTF-8')
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
    #    parser.add_argument('--model_name', type=str, default='simple-u', metavar='N',
    #                        help="network architecture, supporting 'simple-u', 'medium-u', 'ddpm-u'")

    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celeba'], metavar='N',
                        help='dataset used for training (options: cifar10, celeba)')

    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(base_path, 'data') if base_path else '/nfs/fed_diff/data/',
                        help='Data directory')
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

    parser.add_argument('--sampling_steps', type=int, default=100,
                        help='Number of steps to sample from the diffusion model')
    parser.add_argument('--sample_every', type=int, default=500, help='Sample every n steps')
    parser.add_argument('--calculate_fid', action='store_true', help='Calculate FID during training')
    parser.add_argument('--calculate_is', action='store_true', help='Calculate Inception Score during training')
    parser.add_argument('--num_fid_sample', type=int, default=30000,
                        help='Number of samples to use for FID calculation')
    parser.add_argument('--save', action='store_true', help='Save samples during training')
    parser.add_argument('--num_samples', type=int, default=36, help='Number of samples to generate during training')
    parser.add_argument('--cpu_percentage', type=float, default=0.0,
                        help='Percentage of CPU cores to use for data loading')
    parser.add_argument('--fid_estimate_batch_size', type=int, default=32, help='Batch size for FID estimation')
    parser.add_argument('--clip', type=bool, default=True, help='Clip images during generation')
    parser.add_argument('--seed', type=int, default=2024, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps for the scheduler;5000 for cifar10')
    parser.add_argument('--fid_freq', type=int, default=500, help='Frequency of FID calculation')
    parser.add_argument('--central_train', action='store_true', help='Train a centralized model')
    parser.add_argument('--gradient_accumulate_every',type=int, default = 1, help='Number of steps for gradient accumulation')

    # Arguments for FedPhD
    parser.add_argument('--num_edge_servers', type=int, default=2, help='Number of edge servers')
    parser.add_argument('--aggr_freq', type=int, default=5, help='Frequency of aggregation')
    parser.add_argument('--balance_agg_a', type=float, default=5000, help='Balance parameter for aggregation')
    parser.add_argument('--balance_agg_b', type=float, default=0, help='Balance parameter for aggregation')
    parser.add_argument('--sparse_training', action='store_true', help='Enable sparse training')
    parser.add_argument("--train_scratch", action='store_true', help='Use random initialization for training without '
                                                                     'regularization')
    parser.add_argument('--pruning_ratio', type=float, default=0.2, help='Structure pruning ratio')
    # Sparse training is not supported by train from scratch.
    parser.add_argument('--lambda_sparse', type=float, default=0.00001, help='Lambda for sparse training regularization')
    parser.add_argument('--st_rounds', type=int, default=500, help='Intial rounds of sparse training')
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
        'calculate_inception': args.calculate_is,
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
        unet = unet_celeba_standard
        unet.to(args.device)
        # unet_celeba = Unet(dim=128,dim_multiply=(1,2,2,2),image_size=image_size,attn_resolutions=(16,),dropout=0.0,num_res_blocks=2)
        diffusion = GaussianDiffusion(unet, image_size=image_size).to(args.device)
    elif args.dataset == "cifar10":
        image_size = 32
        unet = unet_cifar10_standard
        unet.to(args.device)
        diffusion = GaussianDiffusion(unet, image_size=image_size).to(args.device)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    model = diffusion.to(args.device)
    if out_unet:
        return unet
    else:
        return model


def setup_trainer(args, diffusion_model, fid_scorer, inception_scorer, ddim_samplers, logger):
    # Initialize the trainer with the provided arguments
    trainer = Trainer(args=args, logger=logger,
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

def train_pruned_model(model,args,logger,data_info):
    device = args.device
    group_norm_prune(model=model, args=args, logger=logger)
    if args.dataset == "cifar10":
        image_size = 32
    else:
        image_size = 64
    diffusion_model = GaussianDiffusion(model, image_size=image_size).to(device)
    ddim_samplers = setup_ddim_sampler(args, diffusion_model)  # Just one sampler in defalt
    fid_scorer = setup_fid_scorer(args, image_size=diffusion_model.image_size)
    inception_scorer = setup_inception_scorer(args)
    global_model_trainer = setup_trainer(args, diffusion_model, fid_scorer=fid_scorer,
                                         inception_scorer=inception_scorer, ddim_samplers=ddim_samplers,
                                         logger=logger)
    logger.info(diffusion_model)
    FedPhDAPI = fedphd_api(data_info, device, args, global_model_trainer, logger)
    FedPhDAPI.train()

if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = add_args(argparse.ArgumentParser(description="FedPhD-standalone"))

    args = parser.parse_args()
    device = args.device
    # print("torch version{}".format(torch.__version__))

    Config.initialize(args)

    current_datetime = datetime.now()
    current_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
    data_partition = args.partition_method
    if data_partition != "iid":
        data_partition += str(args.partition_alpha)
    args.identity = "fedphd" + "-" + data_partition
    args.client_num_per_round = int(args.client_num_in_total * args.frac)

    args.identity += "-lambda" + str(args.lambda_sparse)
    args.identity += (
            "-cm" + str(args.comm_round) + "-total_clnt" + str(args.client_num_in_total)
    )
    args.identity += "-neighbor" + str(args.client_num_per_round)
    args.identity += '-balance_agg' + str(args.balance_agg_a)
    args.identity += "-seed" + str(args.seed)
    if args.train_scratch:
        args.identity += "-train_from_begin"
    if args.sparse_training:
        args.identity += "-sparse_training"

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


    if args.dataset == "cifar10":
        data_info = partition_data_indices_cifar10(datadir=args.data_dir, partition=args.partition_method,
                                                   n_nets=args.client_num_in_total, n_cls=args.partition_alpha)
    elif args.dataset == "celeba":
        data_info = partition_data_indices_celeba(datadir=args.data_dir, partition=args.partition_method,
                                                  n_nets=args.client_num_in_total)
    else:
        raise ValueError("Dataset not supported")

    if data_info is None or len(data_info) != 3:
        raise ValueError("Partitioning returned invalid data.")
    if args.train_scratch:
        print("Training from scratch")
        model = load_model(args, out_unet=True)
        train_pruned_model(model,args,logger,data_info)
    else:
        # Sparse training and then fine-tune the pruned model
        # ensure sparse_train is True in your args
        diffusion_model = load_model(args)
        diffusion_model=diffusion_model.to(device)
        num_params = num_params(diffusion_model)
        logger.info("Model num of params:{}".format(num_params))
        # Initail sparse training
        logger.info(f"Sparse training {args.st_rounds} rounds")
        total_rounds = copy.deepcopy(args.comm_round)
        rest_rounds = total_rounds - args.st_rounds
        # set communication round to st_rounds
        args.comm_round = args.st_rounds
        ddim_samplers = setup_ddim_sampler(args, diffusion_model)  # Just one sampler in defalt
        fid_scorer = setup_fid_scorer(args, image_size=diffusion_model.image_size)
        inception_scorer = setup_inception_scorer(args)
        global_model_trainer = setup_trainer(args, diffusion_model, fid_scorer=fid_scorer, ddim_samplers=ddim_samplers,
                                             inception_scorer=inception_scorer, logger=logger)
        logger.info(diffusion_model)

        FedPhDAPI_initial = fedphd_api(data_info, device, args, global_model_trainer, logger)
        sparse_model = FedPhDAPI_initial.train()
        model = load_model(args, out_unet=True) # random init model
        model = model.to('cpu')
        unet_state_dict = {k.replace('unet.', ''): v for k, v in sparse_model.items() if k.startswith('unet.')}
        print('Load sparse model state dict')
        model.load_state_dict(unet_state_dict)
        model.to(device)
        # De-active the sparse training
        args.sparse_training = False
        args.comm_round = rest_rounds

        train_pruned_model(model,args,logger,data_info)


