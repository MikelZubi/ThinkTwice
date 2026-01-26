#!/usr/bin/env bash
#SBATCH -p genoa


#SBATCH --job-name=MUC_Train
#SBATCH --cpus-per-task=64
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
##SBATCH --output=SLURM/SFT-%a.log
##SBATCH --error=SLURM/SFT-%a.err
#SBATCH --output=SLURM/SFT.log
#SBATCH --error=SLURM/SFT.err
#SBATCH --time=20:00:00
#SBATCH --exclude=cn8010
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --exclusive


source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate

export PYTHONPATH=~/DocIE
export NCCL_NET_DISABLE_INTRA=1
export WANDB_MODE=offline
export WANDB_API_KEY=$(cat wandbkey.txt)
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --base-model DeepSeek-R1-Distill-Llama-70B --model-path /leonardo_work/EUHPC_E04_042/BaseModels --output-dir /leonardo_scratch/large/userexternal/mzubilla/TrainedModels --reasoning

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=9000
NUM_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
NUM_GPUS=$(($(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)*4))
MIXED_PRECISION="bf16"
VALUE=8
ITER=$1
mkdir -p "$SCRATCH/SFT/QWEN/$ITER"

srun accelerate launch  --num_processes $NUM_GPUS \
                        --num_machines $NUM_NODES \
                        --mixed_precision "$MIXED_PRECISION" \
                        --dynamo_backend "no" \
                        --multi_gpu \
                        --rdzv_backend c10d \
                        --main_process_ip $MAIN_PROCESS_IP \
                        --main_process_port $MASTER_PORT \
                        --machine_rank $SLURM_NODEID \
                        scripts/MUC_Scripts/trainer/SFT.py \
                        --base-model Qwen3-32B \
                        --model-path $SCRATCH/Ereduak/ \
                        --out-dir "$SCRATCH/SFT/QWEN/$ITER/" \
                        --n $VALUE \
                        --sampling \
                        --epochs 5 \
                        --batch-size 4 \

#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --reasoning
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --reasoning --natural-reasoning