#!/bin/bash

python /work/Software/mace/scripts/run_train.py \
    --name="MACE_A" \
    --seed=111 \
    --train_file="../low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz" \
    --valid_file="../test_all_ITER4.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --default_dtype="float32" \
    --r_max=4.5 \
    --batch_size=10 \
    --max_num_epochs=200 \
    --swa \
    --start_swa=150 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --device=cuda \
    --energy_key=DFT_energy \
    --forces_key=DFT_forces

python /work/Software/mace/scripts/run_train.py \
    --name="MACE_B" \
    --seed=222 \
    --train_file="../low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz" \
    --valid_file="../test_all_ITER4.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --default_dtype="float32" \
    --r_max=4.5 \
    --batch_size=10 \
    --max_num_epochs=200 \
    --swa \
    --start_swa=150 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --device=cuda \
    --energy_key=DFT_energy \
    --forces_key=DFT_forces 

python /work/Software/mace/scripts/run_train.py \
    --name="MACE_C" \
    --seed=333 \
    --train_file="../low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz" \
    --valid_file="../test_all_ITER4.xyz" \
    --config_type_weights='{"Default":1.0}' \
    --model="MACE" \
    --hidden_irreps='128x0e + 128x1o' \
    --default_dtype="float32" \
    --r_max=4.5 \
    --batch_size=10 \
    --max_num_epochs=200 \
    --swa \
    --start_swa=150 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --device=cuda \
    --energy_key=DFT_energy \
    --forces_key=DFT_forces 
