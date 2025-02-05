#!/bin/bash -x
#PJM -L rscgrp=cx-small
#PJM -L node=4
#PJM -L elapse=2:00:00
#PJM -j
#PJM -S
#PJM -o run_system-sys_ppo.job.out

module load gcc/11.3.0 cuda/12.1.1
module load cudnn openmpi_cuda nccl

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OPENAI_API_KEY="<YOUR_API_KEY>"

. .venv/bin/activate

sys_name=sys_ppo

mpirun -n 16 -machinefile $PJM_O_NODEINF -display-devel-map -map-by ppr:2:socket \
    python run_simulation.py \
        --ddp_type mpi \
        --num_sample_turns 10000 \
        --run_dpath outputs/init_data/${sys_name}-10k \
        --random_seed 1 \
        --system_name ${sys_name}
