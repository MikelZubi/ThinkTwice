SPLIT="test"
MODELNAMES=("Llama3.3-70B" "LlamaR1-70B" "Qwen3-32B_nothink" "Qwen3-32B_think")
mkdir -p results/BETTER/zeroshot/${SPLIT}/en_string/voterf1/
for MODELNAME in "${MODELNAMES[@]}"; do
    echo "Processing model: ${MODELNAME}"
    READ_FILE="results/BETTER/zeroshot/${SPLIT}/en_string/${MODELNAME}_64.jsonl"
    OUT_DIR="results/BETTER/zeroshot/${SPLIT}/en_string/voterf1/${MODELNAME}_64.jsonl"
    python scripts/BETTER_Scripts/zeroshot/pseudo_voterf1.py --read ${READ_FILE} --out ${OUT_DIR}
done