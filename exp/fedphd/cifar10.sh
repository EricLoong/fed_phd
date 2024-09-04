#!/bin/bash
python main.py --dataset 'cifar10' \
--partition_method 'noniid-pathological' \
--batch_size 128 \
--lr 2e-4 \
--epochs 1 \
--client_num_in_total 50 --frac 0.2 \
--sample_every 500 \
--comm_round 10200 \
--seed 2023 \
--save \
--calculate_fid \
--calculate_is \
--aggr_freq 5 \
--st_rounds 200 \
--pruning_ratio 0.2 \
--balance_agg_a 50000 \
--num_edge_servers 2 \
--sparse_train \
--tqdm