#!/usr/bin/env bash
#SBATCH -A EUHPC_E04_042
#SBATCH -p boost_usr_prod


#SBATCH --job-name=MUC_Reward_PROBA
#SBATCH --cpus-per-task=32
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
##SBATCH --output=SLURM/SFT-%a.log
##SBATCH --error=SLURM/SFT-%a.err
#SBATCH --output=SLURM/Reward_PROBA_OSOA.log
#SBATCH --error=SLURM/Reward_PROBA_OSOA.err
#SBATCH --time=20:00:00
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --exclusive
##SBATCH --array=0-2


source ~/.bashrc
source $FAST/inguruneak_mikel/DocIE/bin/activate

export PYTHONPATH=~/DocIE
export NCCL_NET_DISABLE_INTRA=1
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --base-model DeepSeek-R1-Distill-Llama-70B --model-path /leonardo_work/EUHPC_E04_042/BaseModels --output-dir /leonardo_scratch/large/userexternal/mzubilla/TrainedModels --reasoning

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=9000
NUM_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
NUM_GPUS=$(($(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)*4))
MIXED_PRECISION="bf16"
#VALUES=(8 16 32)
#VALUE=${VALUES[${SLURM_ARRAY_TASK_ID}]}
VALUE=1
ITER=7
export WANDB_MODE=offline
export WANDB_API_KEY=$(cat wandbkey.txt)

srun accelerate launch  --num_processes $NUM_GPUS \
                        --num_machines $NUM_NODES \
                        --mixed_precision "$MIXED_PRECISION" \
                        --dynamo_backend "no" \
                        --multi_gpu \
                        --rdzv_backend c10d \
                        --main_process_ip $MAIN_PROCESS_IP \
                        --main_process_port $MASTER_PORT \
                        --machine_rank $SLURM_NODEID \
                        scripts/MUC_Scripts/trainer/Reward_proba.py \
                        --base-model Qwen3-Reranker-8B \
                        --model-path $WORK/BaseModels/ \
                        --out-dir "$SCRATCH/Reward/$ITER/" \
                        --epochs 10 \
                        --batch-size 2 \

#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --reasoning
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --reasoning --natural-reasoning