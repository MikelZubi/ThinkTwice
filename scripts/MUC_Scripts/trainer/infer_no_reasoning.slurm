#!/usr/bin/env bash
#SBATCH -p genoa

#SBATCH --job-name=infer_no_reasoning
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=SLURM/infer_gold-dev-%a.log
#SBATCH --error=SLURM/infer_gold-dev-%a.err
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --array=0-17


PATH_LIST=("JSON")
declare -a directory_list
declare -a output_list
declare -a output_list_NoGD
declare -a outpulis_list_train
declare -a output_list_NoGD_train
ITER=SFT-Gold-QWEN
NS=(1 64)
cache="$SCRATCH/tmp/.outlines_MUC_infer_no_reasoning_json_${SLURM_ARRAY_TASK_ID}"
echo $cache
export OUTLINES_CACHE_DIR=$cache
#export VLLM_WORKER_MULTIPROC_METHOD=spawn

for path in "${PATH_LIST[@]}"; do
    basedir="$SCRATCH/SFT/QWEN/$ITER/$path"
    for dir in "$basedir"/*/; do
        dir_name=$(basename "$dir")
        mkdir -p "NoReasoning/QWEN/dev/$ITER"
        OUT="NoReasoning/QWEN/dev/$ITER/$path-$dir_name"
        directory_list+=("$dir")
        output_list+=("$OUT")
    done
done
basedir
OUT="${output_list[${SLURM_ARRAY_TASK_ID}]}"
DIR="${directory_list[${SLURM_ARRAY_TASK_ID}]}"
#DIR="$SCRATCH/SFT/QWEN/$ITER/JSON/checkpoint-99"
#OUT="NoReasoning/QWEN/dev/$ITER/JSON-checkpoint-99"

source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate
#python scripts/MUC_Scripts/trainer/infer_no_reasoning.py --split dev --model-name "$DIR" --out-dir "${OUT}NoGD_1.jsonl" --n 64
python scripts/MUC_Scripts/trainer/infer_no_reasoning.py --split dev --model-name "$DIR" --out-dir "${OUT}_64.jsonl" --n 64 --guided-decoding
#python scripts/MUC_Scripts/trainer/infer_no_reasoning.py --split dev --model-name "$DIR" --out-dir "${OUT}GD_1.jsonl" --guided-decoding