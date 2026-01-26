#!/usr/bin/env bash
#SBATCH -p genoa


#SBATCH --job-name=MUC_Train
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --output=SLURM/select_best_templates.log
#SBATCH --error=SLURM/select_best_templates.err
#SBATCH --time=01:00:00
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus

source ~/.bashrc
source ~/inguruneak/evaluate_iterX/bin/activate

ITER=$1

python scripts/MUC_Scripts/trainer/scoreRejectionSampling.py --iter $ITER --n 8