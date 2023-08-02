#!/bin/bash

python /work/Software/mace/scripts/run_train.py \
    --name="MACE_model" \
    --train_file="low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz" \
    --valid_fraction=0.05 \
    --test_file="test_all_ITER4.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --r_max=6.0 \
    --batch_size=10 \
    --max_num_epochs=1500 \
    --swa \
    --start_swa=1000 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --restart_latest \
    --device=cuda \
    --energy_key=DFT_energy \
    --forces_key=DFT_forces 
