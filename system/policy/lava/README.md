# LAVA
- Paper: [LAVA: Latent Action Spaces via Variational Auto-encoding for Dialogue Policy Optimization](https://aclanthology.org/2020.coling-main.41/)
- Official code: [general/dsml/LAVA - Public](https://gitlab.cs.uni-duesseldorf.de/general/dsml/lava-public)
- ConvLab-3 code: [ConvLab/ConvLab-3/convlab/policy/lava](https://github.com/ConvLab/ConvLab-3/tree/master/convlab/policy/lava)

## Setup
Download some necessary files for LAVA model from the official repo.
1. Preprocessed multiwoz dialogue data
    ```bash
    wget https://gitlab.cs.uni-duesseldorf.de/general/dsml/lava-public/-/raw/master/data.zip
    unzip data.zip
    ```
    This will create a `data` folder containing the preprocessed multiwoz data `norm-multi-woz/train_dials.json`.

2. Fine-tuned LAVA model and config
    ```bash
    wget https://gitlab.cs.uni-duesseldorf.de/general/dsml/lava-public/-/raw/master/experiments_woz/sys_config_log_model/2020-05-12-14-51-49-actz_cat/config.json
    wget https://gitlab.cs.uni-duesseldorf.de/general/dsml/lava-public/-/raw/master/experiments_woz/sys_config_log_model/2020-05-12-14-51-49-actz_cat/rl-2020-05-18-10-50-48/reward_best.model
    ```

Once completed, you can run the LAVA model
