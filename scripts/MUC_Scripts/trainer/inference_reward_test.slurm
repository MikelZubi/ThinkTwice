#!/usr/bin/env bash
#SBATCH -p genoa

#SBATCH --job-name=infererence_reward_GD
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=100GB
#SBATCH --gres=gpu:1
#SBATCH --output=SLURM/infererence_reward-models-test.log
#SBATCH --error=SLURM/infererence_reward-models-test.err
#SBATCH --time=06:00:00
#SBATCH --exclude=cn8001
##SBATCH --array=0-3

export VLLM_USE_V1=0
source ~/.bashrc
source ~/inguruneak/DocIE/bin/activate
declare -a directory_list

#python scripts/MUC_Scripts/trainer/inference_reward.py --split dev --predict rejectionSampling/dev/0/Reasoning_64.jsonl  --model-name "/leonardo_scratch/large/userexternal/mzubilla/Reward/7/checkpoint-1656"  --out-dir rejectionSampling/dev/rewardproba_${CHKPT}/iter0.jsonl
#python scripts/MUC_Scripts/trainer/inference_reward.py --split dev --predict /leonardo/home/userexternal/mzubilla/DocIE/rejectionSampling/dev/7/Sampling_16-checkpoint-163-NoGD_64.jsonl  --model-name "/leonardo_scratch/large/userexternal/mzubilla/Reward/7/checkpoint-1656"  --out-dir rejectionSampling/dev/rewardproba_${CHKPT}/iter7_Sampling_16-checkpoint-163.jsonl
ITER="ALL100GD_QWEN"
#READ_DIRS=("rejectionSampling/dev/DPO/Sampling_1_LORA-checkpoint-780_64.jsonl" "rejectionSampling/dev/DPO/Sampling_1_LORA-checkpoint-780_128.jsonl" "rejectionSampling/dev/iter7comp/JSON_64.jsonl" "rejectionSampling/dev/iter7comp/JSON_128.jsonl" "rejectionSampling/dev/iter7comp/Reasoning_64.jsonl" "rejectionSampling/dev/iter7comp/Reasoning_128.jsonl" "rejectionSampling/dev/0/JSON_64.jsonl" "rejectionSampling/dev/0/JSON_128.jsonl" "rejectionSampling/dev/0/Reasoning_64.jsonl" "rejectionSampling/dev/0/Reasoning_128.jsonl")
#OUT_DIRS=("rejectionSampling/dev/rewardproba_${CHKPT}/DPO_checkpoint-780_64.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/DPO_checkpoint-780_128.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/iter7_JSON_64.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/iter7_JSON_128.jsonl"  "rejectionSampling/dev/rewardproba_${CHKPT}/iter7_Reasoning_64.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/iter7_Reasoning_128.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/iter0_JSON_64.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/iter0_JSON_128.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/iter0_Reasoning_64.jsonl" "rejectionSampling/dev/rewardproba_${CHKPT}/iter0_Reasoning_128.jsonl")


basedir="$SCRATCH/Reward/QWEN/$ITER"
for dir in "$basedir"/*/; do
    dir_name=$(basename "$dir")
    directory_list+=("$dir_name")
done

#DIR="checkpoint-6674"
#DIR="checkpoint-6362"
echo "$DIR"
echo "$SCRATCH/Reward/QWEN/$ITER/$DIR"
mkdir -p "rejectionSampling/QWEN/test/RewardQWEN"

#mkdir -p "rejectionSampling/test/reward_${ITER}"

#python scripts/MUC_Scripts/trainer/inference_reward.py --split test --predict "rejectionSampling/QWEN/test/ALL/Sampling_1-checkpoint-22_64.jsonl" --model-name "/scratch/ehu_p518_1/ehu_p518_1_1/Reward/LLAMA/ALL100GD_QWEN2/checkpoint-3337" --out-dir "rejectionSampling/QWEN/test/RewardQWEN/RLlama_ALL_1-checkpoint-22_64.jsonl" --guidelines
#python scripts/MUC_Scripts/trainer/inference_reward.py --split test --predict "rejectionSampling/QWEN/test/ALL/Sampling_1-checkpoint-22_64.jsonl" --model-name "$SCRATCH/Reward/QWEN/$ITER/$DIR"  --out-dir "rejectionSampling/QWEN/test/RewardQWEN/${DIR}_ALL_1-checkpoint-22_64.jsonl" --guidelines
#python scripts/MUC_Scripts/trainer/inference_reward.py --split dev --predict "rejectionSampling/QWEN/dev/ALL/Sampling_1-checkpoint-33_64.jsonl" --model-name "$SCRATCH/Reward/QWEN/$ITER/$DIR"  --out-dir "rejectionSampling/QWEN/dev/RewardQWEN/ALL_1-checkpoint-33_64.jsonl" --guidelines
#python scripts/MUC_Scripts/trainer/inference_reward.py --split dev --predict "rejectionSampling/QWEN/dev/ALL/Sampling_1-checkpoint-88_64.jsonl" --model-name "$SCRATCH/Reward/QWEN/$ITER/$DIR"  --out-dir "rejectionSampling/QWEN/dev/RewardQWEN/ALL_1-checkpoint-88_64.jsonl" --guidelines

DIR="checkpoint-6674"
python scripts/MUC_Scripts/trainer/inference_reward.py --split test --predict "rejectionSampling/QWEN/test/ALL4/Sampling_4-checkpoint-123_64.jsonl" --model-name "$SCRATCH/Reward/QWEN/$ITER/$DIR"  --out-dir "rejectionSampling/QWEN/test/RewardQWEN/${DIR}_ALL4-checkpoint-123_64.jsonl" --guidelines
#python scripts/MUC_Scripts/trainer/inference_reward.py --split test --predict "rejectionSampling/QWEN/test/14/Sampling_8-checkpoint-164_64.jsonl" --model-name "$SCRATCH/Reward/QWEN/$ITER/$DIR"  --out-dir "rejectionSampling/QWEN/test/RewardQWEN/${DIR}_14-checkpoint-164_64.jsonl" --guidelines
#python scripts/MUC_Scripts/trainer/inference_reward.py --split test --predict "NoReasoning/QWEN/test/SFT-Gold-QWEN/JSON-checkpoint-99_64.jsonl" --model-name "$SCRATCH/Reward/QWEN/$ITER/$DIR"  --out-dir "rejectionSampling/QWEN/test/RewardQWEN/${DIR}_Gold-99_64.jsonl" --guidelines
#python scripts/MUC_Scripts/trainer/inference_reward.py --split dev --predict "NoReasoning/QWEN/dev/SFT-Gold-QWEN/JSON-checkpoint-44_64.jsonl" --model-name "$SCRATCH/Reward/QWEN/$ITER/$DIR"  --out-dir "rejectionSampling/QWEN/dev/RewardQWEN/Gold-44_64.jsonl" --guidelines


