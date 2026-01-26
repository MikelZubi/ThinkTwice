#!/usr/bin/env bash
#SBATCH -p genoa

#SBATCH --job-name=pre_tokenize
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1024GB
#SBATCH --gres=gpu:0
#SBATCH --output=SLURM/pre_tokenize.log
#SBATCH --error=SLURM/pre_tokenize.err
#SBATCH --time=06:00:00


source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate

python scripts/MUC_Scripts/trainer/pre_tokenize_data.py