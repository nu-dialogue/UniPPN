#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=4
#PJM -L elapse=1:00:00
#PJM -j
#PJM -S
#PJM -o run_test-uni_utpg-sys_ppo.job.out

module load gcc/11.3.0 cuda/12.1.1
module load cudnn openmpi_cuda nccl

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENAI_API_KEY="<YOUR_API_KEY>"

. .venv/bin/activate

seed=12345
rl_checkpoint_name="sys_ppo-uniP_utpg_gpt2medium-bs64-gradacc1-lr1e-6_constant-ppr0.9"
sys_name=sys_ppo
modules="nlu dst policy nlg"

mpirun -n 16 -machinefile $PJM_O_NODEINF -display-devel-map -map-by ppr:2:socket \
    python run_simulation.py \
        --random_seed ${seed} \
        --ddp_type mpi \
        --do_test True \
        --run_dpath outputs/test/${rl_checkpoint_name} \
        --system_name ${sys_name} \
        --unippn_policy_model_name outputs/rl_unippn/${rl_checkpoint_name}/ppn/checkpoint-199/policy \
        --unippn_value_model_name outputs/rl_unippn/${rl_checkpoint_name}/ppn/checkpoint-199/value \
        --unippn_max_context_turns 3 \
        --unippn_max_prompt_tokens 256 \
        --unippn_target_modules ${modules}

