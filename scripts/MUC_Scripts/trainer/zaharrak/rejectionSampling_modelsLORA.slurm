#!/usr/bin/env bash
#SBATCH --partition=genoa

#SBATCH --job-name=rejectionSampling_Models_LORA
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=SLURM/rejectionSampling-models-LORA-%a.log
#SBATCH --error=SLURM/rejectionSampling-models-LORA-%a.err
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --array=0-8


PATH_LIST=("Sampling_8_LORA" "Sampling_16_LORA" "Sampling_32_LORA")
declare -a directory_list
declare -a output_list
declare -a output_list_NoGD
NS=(2 4 8 16 32 64)
for path in "${PATH_LIST[@]}"; do
    for dir in "/scratch/ehu_p518_1/ehu_p518_1_1/SFT/$path"/*/; do
        dir_name=$(basename "$dir")
        OUT="rejectionSampling/dev/$path-$dir_name-LORA"
        OUT_NOGD="rejectionSampling/dev/$path-$dir_name-LORA-NoGD"
        directory_list+=("$dir")
        output_list+=("$OUT")
        output_list_NoGD+=("$OUT_NOGD")
    done
done


MERGED_PATH="/scratch/ehu_p518_1/ehu_p518_1_1/SFT/Merged_Model_{$SLURM_ARRAY_TASK_ID}"
OUT="${output_list[${SLURM_ARRAY_TASK_ID}]}"
DIR="${directory_list[${SLURM_ARRAY_TASK_ID}]}"
OUT_NOGD="${output_list_NoGD[${SLURM_ARRAY_TASK_ID}]}"

source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate

python scripts/MUC_Scripts/trainer/unload_LORA.py  --chkpt-path "$DIR" --out-dir "$MERGED_PATH"

for N in "${NS[@]}"; do
    python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n $N --model-name "$MERGED_PATH" --out-dir "$OUT\_$N.jsonl"
    python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split dev --n $N --model-name "$MERGED_PATH" --out-dir "$OUT_NOGD\_$N.jsonl"
done