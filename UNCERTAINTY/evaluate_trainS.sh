#!/bin/bash

python /work/Software/mace/scripts/eval_configs.py \
	--configs="low_F_TRAINING_ITER4_20_MACE.xyz" \
	--model="fitABC/MACE_A_swa.model" \
	--info_prefix="MACE_fitA_" \
	--device=cuda \
	--batch_size=10 \
	--default_dtype="float32" \
	--output="./eval_outs_trainS/trainS_MACE_all_and_ABC.xyz"

python /work/Software/mace/scripts/eval_configs.py \
	--configs="./eval_outs_trainS/trainS_MACE_all_and_ABC.xyz" \
	--model="fitABC/MACE_B_swa.model" \
	--info_prefix="MACE_fitB_" \
	--device=cuda \
	--batch_size=10 \
	--default_dtype="float32" \
	--output="./eval_outs_trainS/trainS_MACE_all_and_ABC.xyz"

python /work/Software/mace/scripts/eval_configs.py \
	--configs="./eval_outs_trainS/trainS_MACE_all_and_ABC.xyz" \
	--model="fitABC/MACE_C_swa.model" \
	--info_prefix="MACE_fitC_" \
	--device=cuda \
	--batch_size=10 \
	--default_dtype="float32" \
	--output="./eval_outs_trainS/trainS_MACE_all_and_ABC.xyz"
