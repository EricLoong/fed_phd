#!/bin/bash
python main.py --dataset 'celeba' \
--partition_method 'noniid-pathological' \
--batch_size 64 \
--lr 2e-5 \
--epochs 1 \
--client_num_in_total 20 --frac 0.2 \
--sample_every 500 \
--comm_round 10000 \
--seed 2023 \
--save \
--calculate_fid \
--calculate_is \
--aggr_freq 5 \
--balance_agg_a 0 \
--num_edge_servers 2 \
--train_scratch \
--tqdm