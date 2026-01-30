LANGUAGES=("en" "ar" "fa" "ko" "ru" "zh" "en_string")
BOOSTRAP_N=$2
language=${LANGUAGES[$1]}
split="test"

MODELS_BETTER=("Llama3.3-70B" "LlamaR1-70B" "Qwen3-32B_nothink" "Qwen3-32B_think")
MODELS_MUC=("Llama3.3-70B" "LlamaR1-70B" "Qwen3-32B_nothink" "Qwen3-32B_think" "MUCLLAMA" "MUCR1" "MUCQWEN_nothink" "MUCQWEN_think")


if [ "$language" == "en_string" ]; then
    mkdir -p results/BETTER/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/voterf1/
    BETTER_Scorer 
    for MODELNAME in "${MODELS_BETTER[@]}"; do
        echo "Processing model: ${MODELNAME}"
        READ_FILE="results/BETTER/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/${MODELNAME}_64.jsonl"
        OUT_DIR="results/BETTER/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/voterf1/${MODELNAME}_1.jsonl"
        python scripts/BETTER_Scripts/zeroshot/pseudo_voterf1.py --read ${READ_FILE} --out ${OUT_DIR}
    done
else
    mkdir -p results/MUC/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/voterf1/
    evaluate_iterX
    for MODELNAME in "${MODELS_MUC[@]}"; do
        echo "Processing model: ${MODELNAME}"
        READ_FILE="results/MUC/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/${MODELNAME}_64.jsonl"
        OUT_DIR="results/MUC/zeroshot/${split}/${language}/boostrapp${BOOSTRAP_N}/voterf1/${MODELNAME}_64.jsonl"
        python scripts/MUC_Scripts/zeroshot/scorer_voterf1.py --read ${READ_FILE} --out ${OUT_DIR}
    done
    python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/$split/$language/voterf1 --split $split --language $language --voterf1