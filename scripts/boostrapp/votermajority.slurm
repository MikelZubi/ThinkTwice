#!/bin/bash
#SBATCH --qos=serial
##SBATCH --partition=serial
##SBATCH --account=hitz-exclusive
#SBATCH --job-name=boostrapp_VoterF1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --output=./SLURM/boostrapp_VoterF1-%a.log
#SBATCH --error=./SLURM/boostrapp_VoterF1-%a.err
#SBATCH --array=0-20

source ~/.bashrc

ID=$SLURM_ARRAY_TASK_ID
INP1=$((ID / 7))
INP2=$((ID % 7))

bash scripts/boostrapp/votermajority.sh $INP1 $INP2
