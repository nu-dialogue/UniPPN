export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

ep=20
bs=16
ga=1
lr=5e-5

torchrun --nnodes 1 --nproc-per-node 8 \
    train.py \
        --model_name_or_path models/pretrained/epoch_6_best_ckpt \
        --text_column input \
        --summary_column target \
        --train_file processed_data/train.json \
        --validation_file processed_data/val.json \
        --additional_vocab_file processed_data/delex_vocab.json \
        --output_dir models/fintuned/ep${ep}_bs${bs}_ga${ga}_lr${lr}/checkpoints \
        --logging_dir models/fintuned/ep${ep}_bs${bs}_ga${ga}_lr${lr}/log \
        --report_to tensorboard \
        --do_train \
        --do_eval \
        --num_train_epochs ${ep} \
        --per_device_train_batch_size ${bs} \
        --per_device_eval_batch_size ${bs} \
        --gradient_accumulation_steps ${ga} \
        --learning_rate ${lr} \
        --logging_strategy steps \
        --logging_steps 100 \
        --eval_strategy steps \
        --eval_steps 500 \
        --save_strategy steps \
        --save_steps 2000
