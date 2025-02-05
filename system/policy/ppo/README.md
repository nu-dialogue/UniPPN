# PPO Policy
A policy model trained with the PPO algorithm. The weights are initialized with the MLE policy model.

## Train
Use the `train_e2e.py` script in this directory if you want to train the policy model on your own.
```bash
repo_dir=$(git rev-parse --show-toplevel)
python train_e2e.py \
    --config_path config.json \
    --load_path ${repo_dir}/ConvLab-2/convlab2/policy/mle/save/best_mle \
    --epoch 50
```
Make sure to set the `--load_path` argument to the path of the pretrained MLE model.

See the `train_e2e.sh` script for more details.

## Setup
Copy trained weights to ConvLab-2 policy model directory.
```bash
repo_dir=$(git rev-parse --show-toplevel)
cp -r save ${repo_dir}/ConvLab-2/convlab2/policy/ppo/
```
