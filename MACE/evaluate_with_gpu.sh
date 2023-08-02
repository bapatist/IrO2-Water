#!/bin/bash

python /work/Software/mace/scripts/eval_configs.py \
	--configs="test_all_ITER4.xyz"\
        --model="MACE_model.model" \
	--device=cuda \
	--batch_size=10 \
        --output="./test_all_I4_MACE_GPU.xyz"
