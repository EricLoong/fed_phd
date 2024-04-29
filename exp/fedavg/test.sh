#!/bin/bash

# Declare an array of epoch values
epochs_arr=(1 2 5)

# Declare an array of client_num_in_total values
clients_arr=(1, 5 10 20)

# Loop over epochs
for epoch in "${epochs_arr[@]}"; do
  # Loop over client numbers
  for client_num in "${clients_arr[@]}"; do
    # Run your script and wait for it to finish before continuing
    python main.py --dataset 'cifar10' \
      --batch_size 128 \
      --lr 2e-4 \
      --epochs $epoch \
      --client_num_in_total $client_num \
      --frac 1 \
      --sample_every 5000 \
      --comm_round 800000 \
      --seed 2023 \
      --save \
      --calculate_fid \
      --tqdm > "epoch_${epoch}_clients_${client_num}.log"
  done
done
