#!/bin/bash
#SBATCH --job-name=4090_causal_mamba
#SBATCH --partition=gpu4090
#SBATCH --qos=2gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/4090_%j.out
#SBATCH --error=slurm_logs/4090_%j.err

export PYTHONPATH=$PYTHONPATH:$(pwd)

# Optimized for Dual 4090 (24GB x 2)
# hidden_dim=128 to ensure stability on 24GB VRAM with 10 layers
# batch_size=16 per GPU (total 32)
python src/training/train.py \
    --data_path /gpfs/work/aac/haiyangjin24/Causal_MambaSA/data/MOSEI/unaligned.pkl \
    --save_dir /gpfs/work/aac/haiyangjin24/Causal_MambaSA/output_4090 \
    --hidden_dim 128 \
    --num_intra_layers 6 \
    --num_fusion_layers 4 \
    --batch_size 32 \
    --lr 2e-4 \
    --ortho_weight 0.2 \
    --modality_dropout 0.2 \
    --weight_decay 0.05
