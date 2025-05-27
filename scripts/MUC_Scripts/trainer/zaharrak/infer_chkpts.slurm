#!/usr/bin/env bash
#SBATCH --partition=hitz-exclusive
#SBATCH --account=hitz-exclusive
#SBATCH --job-name=MUC_LLM_CHKPTS
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100-sxm4
#SBATCH --output=SLURM/chpts_select.log
#SBATCH --error=SLURM/chpts_select.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --time=15:00:00


source /scratch/mzubillaga/inguruneak/DocIE/bin/activate

# Define the parent directory (change this as needed)
PARENT_DIR="Model_JSON_LORA_R1"

# Define the Python script to run

# Iterate over each directory inside the parent directory
for dir in "$PARENT_DIR"/*; do
    if [ -d "$dir" ]; then
        python scripts/MUC_Scripts/trainer/infer_chkpts.py --model-path "$dir" --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    fi
done


# Define the parent directory (change this as needed)
PARENT_DIR="Model_Reasoning_LORA_R1"

# Define the Python script to run

# Iterate over each directory inside the parent directory
for dir in "$PARENT_DIR"/*; do
    if [ -d "$dir" ]; then
        python scripts/MUC_Scripts/trainer/infer_chkpts.py --model-path "$dir" --reasoning --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    fi
done

# Define the parent directory (change this as needed)
PARENT_DIR="Model_Natural_Reasoning_LORA_R1"

# Define the Python script to run

# Iterate over each directory inside the parent directory
for dir in "$PARENT_DIR"/*; do
    if [ -d "$dir" ]; then
        python scripts/MUC_Scripts/trainer/infer_chkpts.py --model-path "$dir" --reasoning --natural-reasoning --model-name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    fi
done

# Define the parent directory (change this as needed)
PARENT_DIR="Model_JSON_LORA"

# Define the Python script to run

# Iterate over each directory inside the parent directory
for dir in "$PARENT_DIR"/*; do
    if [ -d "$dir" ]; then
        python scripts/MUC_Scripts/trainer/infer_chkpts.py --model-path "$dir"
    fi
done


# Define the parent directory (change this as needed)
PARENT_DIR="Model_Reasoning_LORA"

# Define the Python script to run

# Iterate over each directory inside the parent directory
for dir in "$PARENT_DIR"/*; do
    if [ -d "$dir" ]; then
        python scripts/MUC_Scripts/trainer/infer_chkpts.py --model-path "$dir" --reasoning
    fi
done

# Define the parent directory (change this as needed)
PARENT_DIR="Model_Natural_Reasoning_LORA"

# Define the Python script to run

# Iterate over each directory inside the parent directory
for dir in "$PARENT_DIR"/*; do
    if [ -d "$dir" ]; then
        python scripts/MUC_Scripts/trainer/infer_chkpts.py --model-path "$dir" --reasoning --natural-reasoning
    fi
done


(
source /home/mzubillaga/.bashrc
conda activate evaluate_iterX
python scripts/MUC_Scripts/trainer/calculate_results_chkpts.py
)
(
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
python scripts/MUC_Scripts/trainer/load_best_checkpoint.py --r1
python scripts/MUC_Scripts/trainer/load_best_checkpoint.py 
)