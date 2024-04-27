#!/bin/bash
python main.py --dataset 'cifar10' \
--batch_size 128 \
--lr 2e-4 \
--epochs 1 \
--client_num_in_total 1 --frac 1 \
--sample_every 2000 \
--comm_round 800000 \
--seed 2023 \
--save \
--calculate_fid \
--tqdm