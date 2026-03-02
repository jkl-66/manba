#!/bin/bash
#SBATCH --job-name=causal_mamba
#SBATCH --partition=gpu4090
#SBATCH --qos=2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.err

# Load modules and activate environment
# module load anaconda/3
# source activate causal_mamba
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Training command
# Using batch_size 64 for 4090 (24GB) to avoid OOM
python src/training/train.py \
    --data_path /gpfs/work/aac/haiyangjin24/Causal_MambaSA/data/MOSEI/unaligned.pkl \
    --save_dir /gpfs/work/aac/haiyangjin24/Causal_MambaSA/output \
    --batch_size 64 \
    --epochs 100 \
    --lr 2e-4 \
    --hidden_dim 512 \
    --cf_weight 1.0 \
    --modality_dropout 0.1 \
    --num_workers 16
