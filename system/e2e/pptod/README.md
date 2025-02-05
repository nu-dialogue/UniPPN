# Plug-and-Play Task-Oriented Dialogue System (PPTOD)
- Paper: [Multi-Task Pre-Training for Plug-and-Play Task-Oriented Dialogue System](https://arxiv.org/abs/2109.14739)
- Official code: [awslabs/pptod](https://github.com/awslabs/pptod)

You can load and use the released model by running the following code:
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained('ohashi56225/pptod-multiwoz')
tokenizer = T5Tokenizer.from_pretrained('ohashi56225/pptod-multiwoz')
```

Or you can fine-tune the model using the following instructions.

## Fine-tuning
1. Prepare data.
    ```bash
    python prepare_dataset.py --output_dir processed_data
    ```

2. Download pretrained weights from official repo.
    ```bash
    mkdir -p models/pretrained
    cd models/pretrained
    wget https://pptod.s3.amazonaws.com/E2E/epoch_6_best_ckpt.zip
    unzip epoch_6_best_ckpt.zip
    ```

3. Run fine-tuning script.
    ```bash
    . ./train.sh
    ```
