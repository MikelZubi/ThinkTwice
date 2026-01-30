#!/bin/bash
#SBATCH --qos=serial
##SBATCH --partition=serial
##SBATCH --account=hitz-exclusive
#SBATCH --job-name=boostrapp_Reward
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=./SLURM/boostrapp/Reward-%a.log
#SBATCH --error=./SLURM/boostrapp/Reward-%a.err
#SBATCH --array=0-17

source ~/.bashrc

ID=$SLURM_ARRAY_TASK_ID
INP1=$((ID / 6))
INP2=$((ID % 6))

echo "Running boostrapped_reward for language index $INP1 and boostrapp number $INP2"

bash scripts/boostrapp/boostrapped_reward.sh $INP1 $INP2