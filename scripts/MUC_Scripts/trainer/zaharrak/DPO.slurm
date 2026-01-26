#!/usr/bin/env bash
#SBATCH -A EUHPC_E04_042
#SBATCH -p boost_usr_prod


#SBATCH --job-name=MUC_Train
#SBATCH --cpus-per-task=32
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
##SBATCH --output=SLURM/DPO-%a.log
##SBATCH --error=SLURM/DPO-%a.err
#SBATCH --output=SLURM/DPO.log
#SBATCH --error=SLURM/DPO.err
#SBATCH --time=24:00:00
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --exclusive
##SBATCH --array=0-2


source ~/.bashrc
source $FAST/inguruneak_mikel/DocIE/bin/activate

export PYTHONPATH=~/DocIE
export NCCL_NET_DISABLE_INTRA=1
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/DPO.py --base-model DeepSeek-R1-Distill-Llama-70B --model-path /leonardo_work/EUHPC_E04_042/BaseModels --output-dir /leonardo_scratch/large/userexternal/mzubilla/TrainedModels --reasoning

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=9000
NUM_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
NUM_GPUS=$(($(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)*4))
MIXED_PRECISION="bf16"
#VALUES=(8 16 32)
#VALUE=${VALUES[${SLURM_ARRAY_TASK_ID}]}
VALUE=1
ITER=DPO
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
                        scripts/MUC_Scripts/trainer/DPO.py \
                        --base-model DeepSeek-R1-Distill-Llama-70B \
                        --model-path $WORK/BaseModels/ \
                        --out-dir "$SCRATCH/DPO/$ITER/" \
                        --n $VALUE \
                        --epochs 5 \
                        --batch-size 2 \
                        --lora \

#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/DPO.py
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/DPO.py --reasoning
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/DPO.py --reasoning --natural-reasoning