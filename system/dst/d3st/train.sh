export CUDA_VIDIBLE_DEVICES="0,1,2,3"
. ../../../.venv/bin/activate

ep=5
bs=4
lr=5e-5

torchrun --nnodes=1 --nproc-per-node 4 \
    train.py \
        --seed 42 \
        --model_name_or_path google-t5/t5-base \
        --output_dir models/ep${ep}-bs${bs}-lr${lr}/checkpoints \
        --logging_dir models/ep${ep}-bs${bs}-lr${lr}/log \
        --overwrite_output_dir \
        --report_to tensorboard \
        --logging_steps 100 \
        --do_train \
        --num_train_epochs ${ep} \
        --learning_rate ${lr} \
        --per_device_train_batch_size ${bs} \
        --save_steps 2000 \
        --do_eval \
        --per_device_eval_batch_size 8 \
        --evaluation_strategy steps \
        --eval_steps 1000 \
        --train_file processed_data/train.json \
        --validation_file processed_data/dev.json \
        --text_column src\
        --summary_column tgt\
        --max_source_length 2048

