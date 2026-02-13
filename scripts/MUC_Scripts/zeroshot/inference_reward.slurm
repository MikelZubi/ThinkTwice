#!/bin/bash

##SBATCH --qos=regular
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=MUC_Reward
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --time=05:00:00
#SBATCH --output=./SLURM/inference-MUC-reward-%a.log
#SBATCH --error=./SLURM/inference-MUC-reward-%a.err
#SBATCH --array=0-5%4

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate

languages=("ar" "en" "fa" "ko" "ru" "zh")
lang=${languages[$SLURM_ARRAY_TASK_ID]}
#lang="en"
split="test"

MODEL_DIR="/data/mzubillaga/Reward/QWEN/checkpoint-6674"

mkdir -p results/MUC/zeroshot/${split}/${lang}/Reward


echo "MUCQWEN think"
OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/MUCQWEN_think-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/MUCQWEN_think_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang

echo "QWEN think"

OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/Qwen3-32B_think-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/Qwen3-32B_think_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang

echo "MUCQWEN nothink"
OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/MUCQWEN_nothink-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/MUCQWEN_nothink_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang

echo "QWEN nothink"
OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/Qwen3-32B_nothink-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/Qwen3-32B_nothink_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang


echo "Llama R1"
MODEL_DIR="/data/mzubillaga/Reward/LLAMA/checkpoint-10011"

OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/LlamaR1-70B-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/LlamaR1-70B_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang

echo "MUCR1"
MODEL_DIR="/data/mzubillaga/Reward/LLAMA/checkpoint-10011"

OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/MUCR1-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/MUCR1_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang


echo "Llama 3.3"
MODEL_DIR="/data/mzubillaga/Reward/LLAMA/checkpoint-3337"

OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/Llama3.3-70B-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/Llama3.3-70B_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang

echo "MUCLLAMA"
MODEL_DIR="/data/mzubillaga/Reward/LLAMA/checkpoint-3337"

OUT_DIR="results/MUC/zeroshot/${split}/${lang}/Reward/MUCLLAMA-Reward_1.jsonl"
PREDICT_DIR="results/MUC/zeroshot/${split}/${lang}/MUCLLAMA_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference_reward.py --model-dir $MODEL_DIR --predict-dir $PREDICT_DIR --out-dir $OUT_DIR --language $lang