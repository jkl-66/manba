#!/bin/bash
#SBATCH --job-name=a800_causal_mamba
#SBATCH --partition=gpua800
#SBATCH --qos=1a800
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/a800_%j.out
#SBATCH --error=slurm_logs/a800_%j.err

export PYTHONPATH=$PYTHONPATH:$(pwd)

# A800 Optimized Run
# hidden_dim=1024, batch_size=128 (Large models benefit from smaller batches with high-dim)
python src/training/train.py \
    --data_path /gpfs/work/aac/haiyangjin24/Causal_MambaSA/data/MOSEI/unaligned.pkl \
    --save_dir /gpfs/work/aac/haiyangjin24/Causal_MambaSA/output_a800 \
    --hidden_dim 1024 \
    --batch_size 128 \
    --lr 2e-4 \
    --ortho_weight 0.2 \
    --modality_dropout 0.2 \
    --weight_decay 0.05
