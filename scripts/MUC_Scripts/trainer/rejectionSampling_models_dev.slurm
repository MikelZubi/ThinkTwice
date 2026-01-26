#!/usr/bin/env bash
#SBATCH -p genoa

#SBATCH --job-name=rejectionSampling_Models
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=SLURM/rejectionSampling-models-dev-%A-%a.log
#SBATCH --error=SLURM/rejectionSampling-models-dev-%A-%a.err
#SBATCH --time=24:00:00
#SBATCH --exclude=cn8001
#SBATCH --exclusive
#SBATCH --array=0-4


PATH_LIST=("Sampling_8")
declare -a directory_list
declare -a output_list
declare -a output_list_NoGD
declare -a outpulis_list_train
declare -a output_list_NoGD_train
ITER=$1
NS=(64)
cache="$SCRATCH/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
echo $cache
export OUTLINES_CACHE_DIR=$cache
modelname=QWEN
#export VLLM_WORKER_MULTIPROC_METHOD=spawn

for path in "${PATH_LIST[@]}"; do
    basedir="$SCRATCH/SFT/$modelname/$ITER/$path"
    for dir in "$basedir"/*/; do
        dir_name=$(basename "$dir")
        mkdir -p "rejectionSampling/$modelname/dev/$ITER"
        OUT="rejectionSampling/$modelname/dev/$ITER/$path-$dir_name"
        OUT_NOGD="rejectionSampling/$modelname/dev/$ITER/$path-$dir_name-NoGD"
        mkdir -p "rejectionSampling/$modelname/train/$ITER"
        OUT_TRAIN="rejectionSampling/$modelname/train/$ITER/$path-$dir_name"
        OUT_NOGD_TRAIN="rejectionSampling/$modelname/train/$ITER/$path-$dir_name-NoGD"
        directory_list+=("$dir")
        output_list+=("$OUT")
        output_list_NoGD+=("$OUT_NOGD")
        outpulis_list_train+=("$OUT_TRAIN")
        output_list_NoGD_train+=("$OUT_NOGD_TRAIN")

    done
done

OUT="${output_list[${SLURM_ARRAY_TASK_ID}]}"
DIR="${directory_list[${SLURM_ARRAY_TASK_ID}]}"
OUT_NOGD="${output_list_NoGD[${SLURM_ARRAY_TASK_ID}]}"
OUT_TRAIN="${outpulis_list_train[${SLURM_ARRAY_TASK_ID}]}"
OUT_NOGD_TRAIN="${output_list_NoGD_train[${SLURM_ARRAY_TASK_ID}]}"
DIR_TRAIN="${directory_list[${SLURM_ARRAY_TASK_ID}]}"

source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate

echo $DIR
#python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n 1 --model-name "$DIR" --out-dir "${OUT}_1.jsonl" --think
for N in "${NS[@]}"; do
  echo "N=$N"
    python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n $N --model-name "$DIR" --out-dir "${OUT}_${N}.jsonl"
    #python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split dev --n $N --out-dir "${OUT_NOGD}_${N}.jsonl" --model-name "$DIR" --think
    #python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split dev --n $N --model-name "$DIR" --out-dir "${OUT_NOGD}_${N}_W2.jsonl" --add-wait 2
    #python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n $N --model-name "$DIR" --out-dir "${OUT}_${N}_W2.jsonl" --add-wait 2
    #python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n $N --model-name "$DIR" --out-dir "${OUT}_${N}.jsonl"
    #if [ "$SLURM_ARRAY_TASK_ID" -gt 8 ]; then #TODO GEO BORRATU!
    #  python scripts/MUC_Scripts/trainer/rejectionSampling.py --split train --n $N --model-name "$DIR_TRAIN" --out-dir "${OUT_TRAIN}_${N}.jsonl"
    #fi
    #python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split train --n $N --model-name "$DIR_TRAIN" --out-dir "${OUT_NOGD_TRAIN}_${N}.jsonl"
done