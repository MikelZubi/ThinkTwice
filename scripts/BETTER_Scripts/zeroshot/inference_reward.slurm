#!/bin/bash

##SBATCH --qos=regular
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=BETTER_LLM
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=12:00:00
#SBATCH --output=./SLURM/inference-BETTER-reward.log
#SBATCH --error=./SLURM/inference-BETTER-reward.err

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate

MODEL_DIR="/scratch/mzubillaga/Reward/checkpoint-3337"
OUT_DIR="results/BETTER/zeroshot/dev/en_string/LlamaR1-Reward_1.jsonl"
PREDICT_DIR="results/BETTER/zeroshot/dev/en_string/LlamaR1-70B_64.jsonl"

python scripts/BETTER_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language en
