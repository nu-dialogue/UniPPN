#!/bin/bash -x
#PJM -L rscgrp=cx-single
#PJM -L node=1
#PJM -L elapse=12:00:00
#PJM -j
#PJM -S
#PJM -o run_il-uni_ut-sys_ppo-ppr0.8.job.out

export CUDA_VIDIBLE_DEVICES="0,1,2,3"

. .venv/bin/activate
. utils/modules2id.sh

sys_name=sys_ppo
modules="nlu dst policy nlg"

pp_ratio=0.9
spt=5

# 1 Train only embeddings
epochs=1
bs=8
ga=2
lr=5e-3
torchrun --nnodes=1 --nproc-per-node 4 \
    run_il.py \
        --seed 12 \
        --model_name_or_path openai-community/gpt2-medium \
        --torch_dtype float32 \
        --output_dir outputs/il/${sys_name}-10k/unippn_$(modules2id "$modules")-gpt2medium-emb-ep${epochs}-ppr${pp_ratio}-spt${spt}/checkpoints \
        --logging_dir outputs/il/${sys_name}-10k/unippn_$(modules2id "$modules")-gpt2medium-emb-ep${epochs}-ppr${pp_ratio}-spt${spt}/log \
        --overwrite_output_dir \
        --report_to tensorboard \
        --logging_steps 100 \
        --init_data_dir outputs/init_data/${sys_name}-10k \
        --ppn_target_modules ${modules} \
        --train_only_embeddings True \
        --num_turns 10000 \
        --num_samples_per_turn ${spt} \
        --only_successful_dialogue False \
        --postprocessing_ratio ${pp_ratio} \
        --validation_split_ratio 0.2 \
        --do_train \
        --num_train_epochs ${epochs} \
        --per_device_train_batch_size ${bs} \
        --gradient_accumulation_steps ${ga} \
        --learning_rate ${lr} \
        --save_steps 2000 \
        --do_eval \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy steps \
        --eval_steps 500


# 2 Train all parameters
epochs=1
bs=8
ga=2
lr=5e-5
torchrun --nnodes=1 --nproc-per-node 4 \
    run_il.py \
        --seed 123 \
        --model_name_or_path outputs/il/${sys_name}-10k/unippn_$(modules2id "$modules")-gpt2medium-emb-ep1-ppr${pp_ratio}-spt${spt}/checkpoints \
        --torch_dtype float32 \
        --output_dir outputs/il/${sys_name}-10k/unippn_$(modules2id "$modules")-gpt2medium-ep${epochs}-ppr${pp_ratio}-spt${spt}/checkpoints \
        --logging_dir outputs/il/${sys_name}-10k/unippn_$(modules2id "$modules")-gpt2medium-ep${epochs}-ppr${pp_ratio}-spt${spt}/log \
        --overwrite_output_dir \
        --report_to tensorboard \
        --logging_steps 100 \
        --init_data_dir outputs/init_data/${sys_name}-10k \
        --ppn_target_modules ${modules} \
        --train_only_embeddings False \
        --num_turns 10000 \
        --num_samples_per_turn ${spt} \
        --only_successful_dialogue False \
        --postprocessing_ratio ${pp_ratio} \
        --validation_split_ratio 0.2 \
        --do_train \
        --num_train_epochs ${epochs} \
        --per_device_train_batch_size ${bs} \
        --gradient_accumulation_steps ${ga} \
        --learning_rate ${lr} \
        --save_steps 2000 \
        --do_eval \
        --per_device_eval_batch_size 32 \
        --evaluation_strategy steps \
        --eval_steps 500
