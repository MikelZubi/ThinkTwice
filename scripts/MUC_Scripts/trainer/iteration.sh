#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: ITER argument not provided"
  exit 1
fi

ITER=$1
echo "Starting iteration $ITER"

# Submit the first job and extract its job ID
JOBID=$(sbatch scripts/MUC_Scripts/trainer/SFT.slurm $ITER | awk '{print $4}')
echo "First job submitted with Job ID: $JOBID"
# Submit the second job with dependency on the first
#sbatch --dependency=afterok:$JOBID scripts/MUC_Scripts/trainer/rejectionSampling_models_train128.slurm $ITER
sbatch --dependency=afterok:$JOBID scripts/MUC_Scripts/trainer/rejectionSampling_models_train.slurm $ITER
sbatch --dependency=afterok:$JOBID scripts/MUC_Scripts/trainer/rejectionSampling_models_dev.slurm $ITER