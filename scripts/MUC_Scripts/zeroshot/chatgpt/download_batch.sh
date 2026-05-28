
export OPENAI_API_KEY="$(cat /scratch/mzubillaga/inguruneak/OPENAI_API_KEY.txt)"
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
#languages=("ar"  "fa" "ko" "ru" "zh")
languages=("ar")
#split="test"
split="test"
n=1
for lang in "${languages[@]}"; do
    OUT_DIR="results/MUC/zeroshot/$split/$lang/gpt-5.5_think_1.jsonl"
    mkdir -p $(dirname $OUT_DIR)
    BATCH_FILE="/scratch/mzubillaga/tmp/MUC_batch_${lang}_id.txt"
    python scripts/MUC_Scripts/zeroshot/chatgpt/download_batch.py --language $lang --n $n --split $split --batch-id-file $BATCH_FILE --think --out-dir $OUT_DIR
done

