#!/usr/bin/env bash
#SBATCH -p genoa

#SBATCH --job-name=rejectionSampling_Models_NoGD
#SBATCH --cpus-per-task=64
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=SLURM/rejectionSampling-models-test.log
#SBATCH --error=SLURM/rejectionSampling-models-test.err
#SBATCH --time=24:00:00
#SBATCH --exclusive
##SBATCH --array=0-9


ITER=ALL4
NS=(64)
cache="$SCRATCH/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
echo $cache
export OUTLINES_CACHE_DIR=$cache
#export VLLM_WORKER_MULTIPROC_METHOD=spawn

#DIR="$SCRATCH/SFT/QWEN/${ITER}/Sampling_1/checkpoint-22"
#OUT="rejectionSampling/QWEN/test/${ITER}/Sampling_1-checkpoint-22"
DIR="$SCRATCH/SFT/QWEN/ALL4/Sampling_4/checkpoint-123"
OUT="rejectionSampling/QWEN/test/ALL4/Sampling_4-checkpoint-123"
#OUT_NOGD="rejectionSampling/QWEN/test/${ITER}/Sampling_1-checkpoint-22-NoGD"
mkdir -p rejectionSampling/QWEN/test/ALL4
source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate
echo $DIR
python scripts/MUC_Scripts/trainer/rejectionSampling.py --split test --n 1 --model-name "$DIR" --out-dir "${OUT}_1.jsonl" --think
for N in "${NS[@]}"; do
    python scripts/MUC_Scripts/trainer/rejectionSampling.py --split test --n $N --model-name "$DIR" --out-dir "${OUT}_${N}.jsonl" --think
    #python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split test --n $N --out-dir "${OUT_NOGD}_${N}.jsonl" --model-name "$DIR"
    #python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split dev --n $N --model-name "$DIR" --out-dir "${OUT_NOGD}_${N}_W2.jsonl" --add-wait 2
    #python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n $N --model-name "$DIR" --out-dir "${OUT}_${N}_W2.jsonl" --add-wait 2
    #python scripts/MUC_Scripts/trainer/rejectionSampling.py --split dev --n $N --model-name "$DIR" --out-dir "${OUT}_${N}.jsonl"
    #if [ "$SLURM_ARRAY_TASK_ID" -gt 8 ]; then #TODO GEO BORRATU!
    #  python scripts/MUC_Scripts/trainer/rejectionSampling.py --split train --n $N --model-name "$DIR_TRAIN" --out-dir "${OUT_TRAIN}_${N}.jsonl"
    #fi
    #python scripts/MUC_Scripts/trainer/rejectionSampling_NoGD.py --split train --n $N --model-name "$DIR_TRAIN" --out-dir "${OUT_NOGD_TRAIN}_${N}.jsonl"
done