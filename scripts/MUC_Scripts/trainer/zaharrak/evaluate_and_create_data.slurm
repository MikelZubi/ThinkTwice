#!/usr/bin/env bash
#SBATCH -p genoa


#SBATCH --job-name=MUC_Evaluate
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --output=SLURM/evaluate_and_create_data.log
#SBATCH --error=SLURM/evaluate_and_create_data.err
#SBATCH --time=02:00:00
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus

source ~/.bashrc
source ~/inguruneak/evaluate_iterX/bin/activate

ITER=$1

echo "Evaluating"
python scripts/MUC_Scripts/trainer/scoreRejectionSampling.py --iter $ITER --split train

echo "Selecting best templates"
python scripts/MUC_Scripts/trainer/select_best_templates.py --iter $ITER --n 8