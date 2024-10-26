#!/bin/bash
python main.py --dataset 'cifar10' \
--partition_method 'iid' \
--batch_size 128 \
--lr 2e-4 \
--epochs 10 \
--client_num_in_total 10 --frac 1 \
--sample_every 500 \
--comm_round 1000 \
--seed 2023 \
--save \
--calculate_fid \
--tqdm