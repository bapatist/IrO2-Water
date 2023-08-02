#!/bin/bash

python /work/Software/mace/scripts/eval_configs.py \
	--configs="low_F_TRAINING_ITER4_20_WITHOUTGAP.xyz"\
        --model="CPU_MACE_model.model" \
        --output="./low_F_I4_20_MACE.xyz"
