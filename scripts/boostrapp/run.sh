#!/bin/bash
source ~/.bashrc
DocIE

python scripts/boostrapp/boostrapp.py 
sbatch voterf1.slurm
sbatch votermajority.slurm
sbatch boostrapped_reward.slurm