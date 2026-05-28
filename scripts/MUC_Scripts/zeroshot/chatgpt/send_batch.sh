
export OPENAI_API_KEY="$(cat /scratch/mzubillaga/inguruneak/OPENAI_API_KEY.txt)"
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
n=1
languages=("fa" "ko" "ru" "zh")
#languages=("en")
#split="test"
split="test"
for lang in "${languages[@]}"; do
    OUT_DIR="results/MUC/zeroshot/$split/$lang/gpt-5.5_think_1.jsonl"
    BATCH_DIR="/scratch/mzubillaga/tmp/MUC_batch_${lang}.jsonl"
    mkdir -p $(dirname $BATCH_DIR)
    python scripts/MUC_Scripts/zeroshot/chatgpt/prepare_batch.py --model-name gpt-5.5 --language $lang --n $n --split $split --batch-file $BATCH_DIR --think
    python scripts/MUC_Scripts/zeroshot/chatgpt/submit_batch.py --batch-file $BATCH_DIR --think

    BATCH_ID_FILE="/scratch/mzubillaga/tmp/MUC_batch_${lang}_id.txt"
    OUT_DIR="results/MUC/zeroshot/$split/$lang/gpt-5.5_think_1.jsonl"
    while true; do
        STATUS=$(python scripts/MUC_Scripts/zeroshot/chatgpt/check_batch.py --batch-id-file "$BATCH_ID_FILE" | grep "Status:" | awk '{print $2}')
        if [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "cancelled" || "$STATUS" == "expired" ]]; then
            echo "Batch finished with status: $STATUS"
            python scripts/MUC_Scripts/zeroshot/chatgpt/download_batch.py --language $lang --n $n --split $split --batch-id-file $BATCH_ID_FILE --think --out-dir $OUT_DIR
            break
        fi
        echo "Batch status: $STATUS. Waiting 30s..."
        sleep 30
    done
done
