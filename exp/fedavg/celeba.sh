#!/bin/bash
python main.py --dataset 'celeba' \
--partition_method 'noniid-pathological' \
--batch_size 64 \
--lr 2e-4 \
--epochs 5 \
--client_num_in_total 20 --frac 0.2 \
--fid_freq 100 \
--comm_round 1000 \
--seed 2023 \
--save \
--prox \
--gradient_accumulate_every 1 \
--calculate_fid \
--calculate_is \
--tqdm