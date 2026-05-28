
export OPENAI_API_KEY="$(cat /scratch/mzubillaga/inguruneak/OPENAI_API_KEY.txt)"
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate

n=64
split="test"
lang="en"
OUT_DIR="results/BETTER/zeroshot/$split/en_string/gpt-5.5_think_$n.jsonl"
BATCH_DIR="/scratch/mzubillaga/tmp/BETTER_batch_${lang}.jsonl"
mkdir -p $(dirname $BATCH_DIR)
python scripts/BETTER_Scripts/zeroshot/chatgpt/prepare_batch.py --model-name gpt-5.5 --language $lang --n $n --split $split --batch-file $BATCH_DIR --think
python scripts/BETTER_Scripts/zeroshot/chatgpt/submit_batch.py --batch-file $BATCH_DIR --think

BATCH_ID_FILE="/scratch/mzubillaga/tmp/BETTER_batch_${lang}_id.txt"
while true; do
    STATUS=$(python scripts/BETTER_Scripts/zeroshot/chatgpt/check_batch.py --batch-id-file "$BATCH_ID_FILE" | grep "Status:" | awk '{print $2}')
    if [[ "$STATUS" == "completed" || "$STATUS" == "failed" || "$STATUS" == "cancelled" || "$STATUS" == "expired" ]]; then
        echo "Batch finished with status: $STATUS"
        python scripts/BETTER_Scripts/zeroshot/chatgpt/download_batch.py --language $lang --n $n --split $split --batch-id-file $BATCH_ID_FILE --think --out-dir $OUT_DIR
        break
    fi
    echo "Batch status: $STATUS. Waiting 30s..."
    sleep 30
done
