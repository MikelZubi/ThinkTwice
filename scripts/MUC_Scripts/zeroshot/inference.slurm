#!/bin/bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=MUC_LLM
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --gres=gpu:2
#SBATCH --constraint=a100-sxm4
#SBATCH --time=16:00:00
#SBATCH --output=./SLURM/inference-%a.log
#SBATCH --error=./SLURM/inference-%a.err
#SBATCH --array=0-5%2

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
languages=("ar" "en" "fa" "ko" "ru" "zh")


cache="/scratch/mzubillaga/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
export OUTLINES_CACHE_DIR=$cache
lang=${languages[$SLURM_ARRAY_TASK_ID]}
MODEL_NAME="Qwen/Qwen3-32B"
OUT_DIR="results/zeroshot/test/$lang/Qwen3-32B_think_64.jsonl"
mkdir -p $(dirname $OUT_DIR)
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR --think
OUT_DIR="results/zeroshot/test/$lang/Qwen3-32B_think_1.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR --think
OUT_DIR="results/zeroshot/test/$lang/Qwen3-32B_nothink_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR
OUT_DIR="results/zeroshot/test/$lang/Qwen3-32B_nothink_1.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR

MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
OUT_DIR="results/zeroshot/test/$lang/Llama3.3-70B_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR
OUT_DIR="results/zeroshot/test/$lang/Llama3.3-70B_1.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR


MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
OUT_DIR="results/zeroshot/test/$lang/LlamaR1-70B_64.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 64 --split test --out-dir $OUT_DIR --think
OUT_DIR="results/zeroshot/test/$lang/LlamaR1-70B_1.jsonl"
python scripts/MUC_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --language $lang --n 1 --split test --out-dir $OUT_DIR --think
