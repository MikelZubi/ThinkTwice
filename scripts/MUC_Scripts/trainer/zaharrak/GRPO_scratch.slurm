#!/usr/bin/env bash
#SBATCH --qos=xlong
#SBATCH --job-name=MUC_Train_GRPO
#SBATCH --job-name=MUC_Train_GRPO
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=200GB
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100-sxm4
#SBATCH --output=SLURM/GRPO_scratch.log
#SBATCH --error=SLURM/GRPO_scratch.err
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-user=mikel.zubillaga@ehu.eus
#SBATCH --time=8-00:00:00
#geo borratu goikoa
source /home/mzubillaga/.bashrc


#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/GRPO.py --model-path Model_Proba --distributed
#conda activate trainMUC
#export PYTHONPATH=/scratch/mzubillaga/DocIE
#echo "Distributed batch size 2"
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/GRPO.py --model-path Model_JSON --batch-size 2 --distributed
#echo "Distributed batch size 1"
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/GRPO.py --model-path Model_JSON --batch-size 1 --distributed
#echo "Not Distributed batch size 2"
#python scripts/MUC_Scripts/trainer/GRPO.py --model-path Model_JSON --batch-size 2
#echo "Not Distributed batch size 1"
#python scripts/MUC_Scripts/trainer/GRPO.py --model-path Model_JSON --batch-size 1
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/GRPO.py --reasoning --model-path Model_Reasoning
#accelerate launch --main_process_port 29516 scripts/MUC_Scripts/trainer/GRPO.py --reasoning --natural-reasoning --model-path Model_Natural_Reasoning

bash scripts/MUC_Scripts/trainer/GRPO_scratch.sh