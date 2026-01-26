#!/bin/bash

if [ -z "$1" ]; then
  echo "Error: ITER argument not provided"
  exit 1
fi

START_ITER=$1
MAX_ITER=14

PREV_JOBIDE=""

for ITER in $(seq $START_ITER $MAX_ITER); do
  echo "Starting iteration $ITER"

  # Submit SFT job with dependency on previous iteration's evaluation job
  if [ -z "$PREV_JOBIDE" ]; then
    # First iteration without dependency
    JOBIDT=$(sbatch scripts/MUC_Scripts/trainer/SFT.slurm $ITER | awk '{print $4}')
  else
    # Subsequent iterations depend on previous evaluation job
    JOBIDT=$(sbatch --dependency=afterok:$PREV_JOBIDE scripts/MUC_Scripts/trainer/SFT.slurm $ITER | awk '{print $4}')
  fi
  echo "SFT job submitted with Job ID: $JOBIDT"

  # Submit remaining jobs with dependencies
  JOBIDI=$(sbatch --dependency=afterok:$JOBIDT scripts/MUC_Scripts/trainer/rejectionSampling_models_train.slurm $ITER | awk '{print $4}')
  sbatch --dependency=afterok:$JOBIDT scripts/MUC_Scripts/trainer/rejectionSampling_models_dev.slurm $ITER
  JOBIDE=$(sbatch --dependency=afterok:$JOBIDI scripts/MUC_Scripts/trainer/evaluate_and_create_data.slurm $ITER | awk '{print $4}')

  # Store evaluation job ID for next iteration
  PREV_JOBIDE=$JOBIDE
  echo "Iteration $ITER jobs submitted"
done

echo "All iteration jobs submitted"