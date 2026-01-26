#!/usr/bin/env bash
#SBATCH -A EUHPC_E04_042
#SBATCH -p boost_usr_prod


#SBATCH --job-name=MUC_Regresor
#SBATCH --cpus-per-task=32
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=SLURM/Regresor-BERT.log
#SBATCH --error=SLURM/Regresor-BERT.err
#SBATCH --time=10:00:00
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --exclusive
##SBATCH --array=0-2


source ~/.bashrc
source $FAST/inguruneak_mikel/DocIE/bin/activate

export PYTHONPATH=~/DocIE
export NCCL_NET_DISABLE_INTRA=1

# Maximum time (in seconds) to wait for NCCL operations before timing out
#export NCCL_TIMEOUT=14400 # 4 hours, prevent crashing when loading datasets
#export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 # 2 hour, prevent HEARTBEAT_TIMEOUT error
# Enable asynchronous error handling for NCCL operations
#export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Enable the memory allocator to allocate memory in a way that is more likely to be compatible with NCCL
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Trying things until it works. 
# > By the look of it, NCCL decided that communication between the two CPUs on each node will be faster using the NICs than using UPI.
# > However, this won't work if your NICs can't talk to each other...
#
# > You can try setting NCCL_NET_DISABLE_INTRA=1 to prevent the NICs from being used for such purpose.
# (https://github.com/NVIDIA/nccl/issues/1405)
export NCCL_NET_DISABLE_INTRA=1

#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --base-model DeepSeek-R1-Distill-Llama-70B --model-path /leonardo_work/EUHPC_E04_042/BaseModels --output-dir /leonardo_scratch/large/userexternal/mzubilla/TrainedModels --reasoning

MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=9000
NUM_NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)
NUM_GPUS=$(($(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l)*4))
MIXED_PRECISION="bf16" # bf16 beittu a ver modernBertekin ongi dijoan
#VALUES=(8 16 32)
#VALUE=${VALUES[${SLURM_ARRAY_TASK_ID}]}
VALUE=1
SEED=42
echo "Seed: $SEED"
ITER=REGRESOR_BERT
export WANDB_MODE=offline
export WANDB_API_KEY=$(cat wandbkey.txt)

srun accelerate launch  --num_processes $NUM_GPUS \
                        --num_machines $NUM_NODES \
                        --dynamo_backend "no" \
                        --mixed_precision "$MIXED_PRECISION" \
                        --multi_gpu \
                        --rdzv_backend c10d \
                        --main_process_ip $MAIN_PROCESS_IP \
                        --main_process_port $MASTER_PORT \
                        --machine_rank $SLURM_NODEID \
                        scripts/MUC_Scripts/trainer/Regresor.py \
                        --base-model ModernBERT-large \
                        --model-path $WORK/BaseModels/ \
                        --out-dir "$SCRATCH/Regresor/$ITER/" \
                        --epochs 20 \
                        --batch-size 16 \
                        --seed $SEED \

#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --reasoning
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/SFT.py --reasoning --natural-reasoning