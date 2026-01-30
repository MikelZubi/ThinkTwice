LANGUAGES=("en" "ar" "fa" "ko" "ru" "zh")
BOOSTRAP_N=$1
language=${LANGUAGES[$2]}
split="test"

MODELS_MUC=("Llama3.3-70B" "LlamaR1-70B" "Qwen3-32B_nothink" "Qwen3-32B_think" "MUCLLAMA" "MUCR1" "MUCQWEN_nothink" "MUCQWEN_think")

mkdir -p results/MUC/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/Reward/
evaluate_iterX
for MODELNAME in "${MODELS_MUC[@]}"; do
    echo "Processing model: ${MODELNAME}"
    REWARD_FILE="results/MUC/zeroshot/${split}/${language}/Reward/${MODELNAME}_1.jsonl"
    READ_FILE="results/MUC/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/${MODELNAME}_64.jsonl"
    OUT_DIR="results/MUC/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/Reward/${MODELNAME}_64.jsonl"
    python scripts/MUC_Scripts/zeroshot/boostrapped_reward.py --reward_file ${REWARD_FILE} --boostrapp_file ${READ_FILE} --output_file ${OUT_DIR}
done
echo "Boostrapped reward calculation completed for language: ${language}, boostrapp sample: ${BOOSTRAP_N}"