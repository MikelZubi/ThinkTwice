#!/usr/bin/env bash
#SBATCH --partition=genoa

#SBATCH --job-name=rejectionSampling_Models_LORA
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=SLURM/rejectionSampling-models-LORA-train-%a.log
#SBATCH --error=SLURM/rejectionSampling-models-LORA-train-%a.err
#SBATCH --time=2-00:00:00
#SBATCH --exclusive
#SBATCH --array=0-14


PATH_LIST=("Sampling_2_LORA" "Sampling_4_LORA" "Sampling_8_LORA" "Sampling_16_LORA" "Sampling_32_LORA")
declare -a directory_list
declare -a output_list
declare -a output_list_NoGD
declare -a outpulis_list_train
declare -a output_list_NoGD_train
NS=(64)
for path in "${PATH_LIST[@]}"; do
    for dir in "/scratch/ehu_p518_1/ehu_p518_1_1/SFT/$path"/*/; do
        dir_name=$(basename "$dir")
        OUT="rejectionSampling/dev/$path-$dir_name-LORA"
        OUT_NOGD="rejectionSampling/dev/$path-$dir_name-LORA-NoGD"
        OUT_TRAIN="rejectionSampling/train/$path-$dir_name-LORA"
        OUT_NOGD_TRAIN="rejectionSampling/train/$path-$dir_name-LORA-NoGD"
        directory_list+=("$dir")
        output_list+=("$OUT")
        output_list_NoGD+=("$OUT_NOGD")
        outpulis_list_train+=("$OUT_TRAIN")
        output_list_NoGD_train+=("$OUT_NOGD_TRAIN")
    done
done


MERGED_PATH="/scratch/ehu_p518_1/ehu_p518_1_1/SFT/Merged_Model_{$SLURM_ARRAY_TASK_ID}"
OUT="${output_list[${SLURM_ARRAY_TASK_ID}]}"
DIR="${directory_list[${SLURM_ARRAY_TASK_ID}]}"
OUT_NOGD="${output_list_NoGD[${SLURM_ARRAY_TASK_ID}]}"
OUT_TRAIN="${outpulis_list_train[${SLURM_ARRAY_TASK_ID}]}"
OUT_NOGD_TRAIN="${output_list_NoGD_train[${SLURM_ARRAY_TASK_ID}]}"
DIR_TRAIN="${directory_list[${SLURM_ARRAY_TASK_ID}]}"

source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate

python scripts/MUC_Scripts/trainer/unload_LORA.py  --chkpt-path "$DIR" --out-dir "$MERGED_PATH"

for N in "${NS[@]}"; do
    #python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n $N --model-name "$MERGED_PATH" --out-dir "$OUT\_$N.jsonl"
    #python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split dev --n $N --model-name "$MERGED_PATH" --out-dir "$OUT_NOGD\_$N.jsonl"
    if [ "$SLURM_ARRAY_TASK_ID" -gt 5 ]; then #TODO GEO BORRATU!
        python scripts/MUC_Scripts/trainer/rejectionSampling.py --split train --n $N --model-name "$MERGED_PATH" --out-dir "$OUT_TRAIN\_$N.jsonl"
    fi

    python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split train --n $N --model-name "$MERGED_PATH" --out-dir "$OUT_NOGD_TRAIN\_$N.jsonl"
done