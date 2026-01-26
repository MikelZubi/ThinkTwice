#!/bin/bash
#SBATCH --qos=serial
##SBATCH --partition=serial
##SBATCH --account=hitz-exclusive
#SBATCH --job-name=MUC_voterMajority
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --output=./SLURM/voterMajority-%a.log
#SBATCH --error=./SLURM/voterMajority-%a.err
#SBATCH --array=0-5

source ~/.bashrc
source /scratch/mzubillaga/inguruneak/evaluate_iterX/bin/activate
languages=("ar" "en" "fa" "ko" "ru" "zh")
split=test

cache="/scratch/mzubillaga/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
export OUTLINES_CACHE_DIR=$cache
lang=${languages[$SLURM_ARRAY_TASK_ID]}

READ_DIR="results/MUC/zeroshot/$split/$lang/Qwen3-32B_think_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/Qwen3-32B_think_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR

READ_DIR="results/MUC/zeroshot/$split/$lang/Qwen3-32B_nothink_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/Qwen3-32B_nothink_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR

READ_DIR="results/MUC/zeroshot/$split/$lang/Llama3.3-70B_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/Llama3.3-70B_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR


READ_DIR="results/MUC/zeroshot/$split/$lang/LlamaR1-70B_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/LlamaR1-70B_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR

READ_DIR="results/MUC/zeroshot/$split/$lang/MUCR1_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/MUCR1_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR

READ_DIR="results/MUC/zeroshot/$split/$lang/MUCLLAMA_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/MUCLLAMA_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR


READ_DIR="results/MUC/zeroshot/$split/$lang/MUCR1_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/MUCR1_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR

READ_DIR="results/MUC/zeroshot/$split/$lang/MUCLLAMA_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/MUCLLAMA_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR


READ_DIR="results/MUC/zeroshot/$split/$lang/MUCQWEN_think_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/MUCQWEN_think_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR

READ_DIR="results/MUC/zeroshot/$split/$lang/MUCQWEN_nothink_64.jsonl"
OUT_DIR="results/MUC/zeroshot/$split/$lang/voterMajority/MUCQWEN_nothink_64.jsonl"
mkdir -p "$(dirname "$OUT_DIR")"
python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read $READ_DIR --out $OUT_DIR

python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/$split/$lang/voterMajority --split $split --language $lang --voter
#python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/$split/$lang --split $split --language $lang 