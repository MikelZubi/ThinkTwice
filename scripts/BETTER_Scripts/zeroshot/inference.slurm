#!/bin/bash

#SBATCH --qos=regular
##SBATCH --partition=hitz-exclusive
##SBATCH --account=hitz-exclusive
#SBATCH --job-name=BETTER_LLM
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-sxm4
#SBATCH --time=10:00:00
#SBATCH --output=./SLURM/inference-BETTER.log
#SBATCH --error=./SLURM/inference-BETTER.err

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate

SPLIT=test


cache="/scratch/mzubillaga/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
export OUTLINES_CACHE_DIR=$cache
lang=en_string

MODEL_NAME="Qwen/Qwen3-32B"
OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/Qwen3-32B_think_1.jsonl"
mkdir -p $(dirname $OUT_DIR)
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 1 --split $SPLIT --out-dir $OUT_DIR --think
OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/Qwen3-32B_think_64.jsonl"
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 64 --split $SPLIT --out-dir $OUT_DIR --think

OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/Qwen3-32B_nothink_64.jsonl"
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 64 --split $SPLIT --out-dir $OUT_DIR
OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/Qwen3-32B_nothink_1.jsonl"
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 1 --split $SPLIT --out-dir $OUT_DIR

MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/Llama3.3-70B_64.jsonl"
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 64 --split $SPLIT --out-dir $OUT_DIR
OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/Llama3.3-70B_1.jsonl"
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 1 --split $SPLIT --out-dir $OUT_DIR

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/LlamaR1-70B_64.jsonl"
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 64 --split $SPLIT --out-dir $OUT_DIR --think
OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/LlamaR1-70B_1.jsonl"
python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 1 --split $SPLIT --out-dir $OUT_DIR --think


#TRANSFERENTZIA:
#MODEL_NAME="/scratch/mzubillaga/SFT/ALL_R1_1/checkpoint-22"
#OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/MUCR1_think_64.jsonl"
#python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 64 --split $SPLIT --out-dir $OUT_DIR --think
#OUT_DIR="results/BETTER/zeroshot/$SPLIT/$lang/MUCR1_think_1.jsonl"
#python scripts/BETTER_Scripts/zeroshot/inference.py --model-name $MODEL_NAME --n 1 --split $SPLIT --out-dir $OUT_DIR --think