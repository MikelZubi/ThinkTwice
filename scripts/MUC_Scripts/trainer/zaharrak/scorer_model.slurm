#!/usr/bin/env bash
#SBATCH -A EUHPC_E04_042
#SBATCH -p boost_usr_prod

#SBATCH --job-name=scorer-model 
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH --gres=gpu:4
#SBATCH --output=SLURM/scorer-model.log
#SBATCH --error=SLURM/scorer-model.err
#SBATCH --time=1-00:00:00
#SBATCH --exclusive


source ~/.bashrc
cache="/leonardo_scratch/large/userexternal/mzubilla/tmp/.outlines_MUC_rejectionSampling_json_${SLURM_ARRAY_TASK_ID}"
echo $cache
export OUTLINES_CACHE_DIR=$cache
source ~/.bashrc
source $FAST/inguruneak_mikel/DocIE/bin/activate

python scripts/MUC_Scripts/trainer/scorer_model.py --read-file rejectionSampling/dev/5/Sampling_8-checkpoint-328-NoGD_64.jsonl --out-dir rejectionSampling/dev/scorer-proba/zero_shot.jsonl --model-name /leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B --reasoning
#python scripts/MUC_Scripts/trainer/scorer_model.py --read-file multimuc/data/multimuc_v1.0/corrected/en/dev_rejectionSampling_32.jsonl --out-dir rejectionSampling/dev/scorer-proba/zero_shot.jsonl

deactivate
source $FAST/inguruneak_mikel/evaluate_iterX/bin/activate
python scripts/MUC_Scripts/trainer/calculate_results.py 