#!/bin/bash -x
#PJM -L rscgrp=cx-share
#PJM -L elapse=4:00:00
#PJM -j
#PJM -S
#PJM -o train_ppo_policy.job.out

export CUDA_VISIBLE_DEVICES="0"

repo_dir=$(git rev-parse --show-toplevel)

. ${repo_dir}/.venv/bin/activate

python train_e2e.py \
    --config_path config.json \
    --load_path ${repo_dir}/ConvLab-2/convlab2/policy/mle/save/best_mle \
    --epoch 50
