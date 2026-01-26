#!/usr/bin/env bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=Create_CoT
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=130GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
#SBATCH --output=SLURM/create_CoT.log
#SBATCH --error=SLURM/create_CoT.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --time=02:30:00


python scripts/MUC_Scripts/preprocess/MUC_create_CoT.py train
python scripts/MUC_Scripts/preprocess/MUC_create_CoT.py dev