#!/bin/bash
#SBATCH --qos=serial
##SBATCH --partition=serial
##SBATCH --account=hitz-exclusive
#SBATCH --job-name=MUC_RewardAnalysis
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=./SLURM/analysis_reward-%a.log
#SBATCH --error=./SLURM/analysis_reward-%a.err
#SBATCH --array=0-5

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/evaluate_iterX/bin/activate
languages=("ar" "en" "fa" "ko" "ru" "zh")
split=test

language=${languages[$SLURM_ARRAY_TASK_ID]}

python scripts/MUC_Scripts/analysis/reward_correlations.py --language $language --split $split
python scripts/MUC_Scripts/analysis/reward_correlations.py --language $language --split $split --remove-repeated