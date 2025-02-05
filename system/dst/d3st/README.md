# Description-Driven Dialog State Tracking (D3ST)
- Paper: [Description-Driven Task-Oriented Dialog Modeling](https://arxiv.org/abs/2201.08904)
- Code: [google-research/task-oriented-dialogue/state_tracking/d3st](https://github.com/google-research/task-oriented-dialogue/tree/main/state_tracking/d3st)

You can load and use the released model by running the following code:
```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained('ohashi56225/d3st-multiwoz')
tokenizer = AutoTokenizer.from_pretrained('ohashi56225/d3st-multiwoz')
```

Or you can fine-tune the model on MultiWOZ dataset with the following steps.

## Setup
1. Make train/val/test dataset
    ```bash
    python -m create_multiwoz_schemaless_data \
        --multiwoz_dir=../../../dataset/MultiWOZ2_3/ \
        --output_dir=processed_data \
        --schema_file=schema.json \
        --multiwoz_version 2.3 \
        --description_type=full_desc \
        --delimiter== \
        --multiple_choice=none \
        --blocked_domains="bus"
    ```

2. Fine-tune the model
    ```bash
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
    ```