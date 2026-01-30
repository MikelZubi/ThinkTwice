#!/bin/bash
source ~/.bashrc
DocIE

python scripts/boostrapp/boostrapp.py 
sbatch scripts/boostrapp/voterf1.slurm
sbatch scripts/boostrapp/votermajority.slurm
sbatch scripts/boostrapp/boostrapped_reward.slurm