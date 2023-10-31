#!/bin/bash

python /work/Software/mace/scripts/eval_configs.py \
	--configs="md_foo.xyz"\
        --model="fitABC/MACE_A_swa.model" \
	--device=cuda \
	--batch_size=10 \
	--default_dtype="float32" \
	--output="./eval_outs/md_eval_A.xyz"

python /work/Software/mace/scripts/eval_configs.py \
	--configs="md_foo.xyz"\
        --model="fitABC/MACE_B_swa.model" \
	--device=cuda \
	--batch_size=10 \
	--default_dtype="float32" \
	--output="./eval_outs/md_eval_B.xyz"

python /work/Software/mace/scripts/eval_configs.py \
	--configs="md_foo.xyz"\
        --model="fitABC/MACE_C_swa.model" \
	--device=cuda \
	--batch_size=10 \
	--default_dtype="float32" \
	--output="./eval_outs/md_eval_C.xyz"

