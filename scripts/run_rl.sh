#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=4
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S
#PJM -o run_rl-sys_ppo-uniP_utpg-bs64-gradacc1-lr1e6_constant-ppr0.9.job.out

module load gcc/11.3.0 cuda/12.1.1
module load cudnn openmpi_cuda nccl

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENAI_API_KEY="<YOUR_API_KEY>"

. .venv/bin/activate
. utils/modules2id.sh

seed=1234
sys_name=sys_ppo

il_modules="nlu dst policy nlg"
ppn_modules="nlu dst policy nlg"
ppo_modules="nlu dst policy nlg"

bs=64
mbs=8
grad_accum=1
lr=1e-6
schdlr=constant
kl_coef=0.01

ppr=0.9
spt=5
il_train_id="unippn_$(modules2id "$il_modules")-gpt2medium-ep1-ppr${ppr}-spt${spt}"

mpirun -n 16 -machinefile $PJM_O_NODEINF -display-devel-map -map-by ppr:2:socket \
    python run_rl.py \
        --ddp_type mpi \
        --random_seed ${seed} \
        --wandb_project_name ppn-all-${sys_name} \
        --wandb_group_name uniP_$(modules2id "$ppn_modules") \
        --run_dpath outputs/rl_unippn/${sys_name}-uniP_$(modules2id "$ppn_modules")_gpt2medium-bs${bs}-gradacc${grad_accum}-lr${lr}_${schdlr}-ppr${ppr} \
        --system_name ${sys_name} \
        --save_session_logs False \
        --total_iterations 200 \
        --save_iterations 20 \
        --batch_size_per_device ${bs} \
        --unippn_policy_model_name outputs/il/${sys_name}-10k/${il_train_id}/checkpoints \
        --unippn_value_model_name openai-community/gpt2 \
        --unippn_max_context_turns 3 \
        --unippn_max_prompt_tokens 256 \
        --unippn_target_modules ${ppn_modules} \
        --unippo_target_modules_to_train ${ppo_modules} \
        --unippo_policy_train_only_embeddings False \
        --unippo_policy_learning_rate ${lr} \
        --unippo_policy_scheduler ${schdlr} \
        --unippo_minibatch_size_per_device ${mbs} \
        --unippo_gradient_accumulation_steps ${grad_accum} \
        --unippo_policy_kl_penalty_coef ${kl_coef}

