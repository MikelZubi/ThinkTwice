#!/usr/bin/env bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=MUC_LLM_CHKPTS
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --output=SLURM/test_models.log
#SBATCH --error=SLURM/test_models.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --time=00:45:00

source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
#python scripts/MUC_Scripts/trainer/test.py
#python scripts/MUC_Scripts/trainer/test.py --reasoning
python scripts/MUC_Scripts/trainer/test.py --reasoning --natural-reasoning
#python scripts/MUC_Scripts/trainer/test.py --r1
#python scripts/MUC_Scripts/trainer/test.py --r1 --reasoning
#python scripts/MUC_Scripts/trainer/test.py --r1 --reasoning --natural-reasoning