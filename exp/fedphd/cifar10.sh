#!/bin/bash
python main.py --dataset 'cifar10' \
--partition_method 'noniid-pathological' \
--batch_size 128 \
--lr 2e-4 \
--epochs 1 \
--client_num_in_total 20 --frac 0.2 \
--sample_every 500 \
--comm_round 20000 \
--seed 2023 \
--save \
--calculate_fid \
--tqdm