#!/bin/bash

##SBATCH --qos=regular
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=MUC_LLM_crosslingual
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-sxm4
#SBATCH --time=06:00:00
#SBATCH --output=./SLURM/inference-crosslingual-%a.log
#SBATCH --error=./SLURM/inference-crosslingual-%a.err
#SBATCH --array=0-5%1

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
languages=("en" "ar" "fa" "ko" "ru" "zh")
#languages=("ar" "ru")

cache="/scratch/mzubillaga/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
export OUTLINES_CACHE_DIR=$cache
lang=${languages[$SLURM_ARRAY_TASK_ID]}
#MODEL_NAME="/data/mzubillaga/SFT/LLAMA/ALL_1/checkpoint-22"
#OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCR1_64.jsonl"
#mkdir -p $(dirname $OUT_DIR)
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split dev --out-dir $OUT_DIR --think
#OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCR1_1.jsonl"
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split dev --out-dir $OUT_DIR --think
#
#OUT_DIR="results/MUC/zeroshot/test/$lang/MUCR1_64.jsonl"
#mkdir -p $(dirname $OUT_DIR)
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR --think
#OUT_DIR="results/MUC/zeroshot/test/$lang/MUCR1_1.jsonl"
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR --think

MODEL_NAME="/data/mzubillaga/SFT/QWEN/ALL4/checkpoint-123"
#OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCQWEN_think_64.jsonl"
#mkdir -p $(dirname $OUT_DIR)
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split dev --out-dir $OUT_DIR --think
#OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCQWEN_think_1.jsonl"
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split dev --out-dir $OUT_DIR --think

OUT_DIR="results/MUC/zeroshot/test/$lang/MUCQWEN_think_64.jsonl"
mkdir -p $(dirname $OUT_DIR)
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR --think
OUT_DIR="results/MUC/zeroshot/test/$lang/MUCQWEN_think_1.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR --think

#MODEL_NAME="/data/mzubillaga/SFT/LLAMA/SFT-Gold-LLAMA/checkpoint-198"
#OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCLLAMA_64.jsonl"
#mkdir -p $(dirname $OUT_DIR)
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split dev --out-dir $OUT_DIR
#OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCLLAMA_1.jsonl"
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split dev --out-dir $OUT_DIR
#
#OUT_DIR="results/MUC/zeroshot/test/$lang/MUCLLAMA_64.jsonl"
#mkdir -p $(dirname $OUT_DIR)
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR
#OUT_DIR="results/MUC/zeroshot/test/$lang/MUCLLAMA_1.jsonl"
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR


MODEL_NAME="/data/mzubillaga/SFT/QWEN/SFT-Gold-QWEN/checkpoint-99"
OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCQWEN_nothink_64.jsonl"
mkdir -p $(dirname $OUT_DIR)
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split dev --out-dir $OUT_DIR
#OUT_DIR="results/MUC/zeroshot/dev/$lang/MUCQWEN_nothink_1.jsonl"
#mkdir -p $(dirname $OUT_DIR)
#python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split dev --out-dir $OUT_DIR
#OUT_DIR="results/MUC/zeroshot/test/$lang/MUCQWEN_nothink_64.jsonl"
#mkdir -p $(dirname $OUT_DIR)
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR
OUT_DIR="results/MUC/zeroshot/test/$lang/MUCQWEN_nothink_1.jsonl"
mkdir -p $(dirname $OUT_DIR)
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR

