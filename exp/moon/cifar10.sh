#!/bin/bash
python main.py --dataset 'cifar10' \
--partition_method 'noniid-pathological' \
--batch_size 128 \
--lr 2e-4 \
--epochs 5 \
--client_num_in_total 20 --frac 0.2 \
--contrastive_loss_weight 1 \
--sample_every 500 \
--comm_round 10000 \
--seed 2023 \
--save \
--calculate_fid \
--tqdm