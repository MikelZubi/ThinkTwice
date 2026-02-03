#!/bin/bash
#SBATCH --qos=serial
##SBATCH --partition=serial
##SBATCH --account=hitz-exclusive
#SBATCH --job-name=calculate_results
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=./SLURM/calculate_results-%a.log
#SBATCH --error=./SLURM/calculate_results-%a.err
#SBATCH --array=0-5

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/evaluate_iterX/bin/activate
languages=("ar" "en" "fa" "ko" "ru" "zh")
split=test

cache="/scratch/mzubillaga/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
export OUTLINES_CACHE_DIR=$cache
lang=${languages[$SLURM_ARRAY_TASK_ID]}



python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/$split/$lang/voterf1 --split $split --language $lang --voterf1
python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/$split/$lang --split $split --language $lang 
python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/$split/$lang/Reward --split $split --language $lang