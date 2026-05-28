
export OPENAI_API_KEY="$(cat /scratch/mzubillaga/inguruneak/OPENAI_API_KEY.txt)"
source /scratch/mzubillaga/inguruneak/DocIE/bin/activate
n=1
split="test"
lang="en"
OUT_DIR="results/BETTER/zeroshot/$split/en_string/gpt-5.5_think_1.jsonl"
mkdir -p $(dirname $OUT_DIR)
BATCH_FILE="/scratch/mzubillaga/tmp/BETTER_batch_${lang}_id.txt"
python scripts/BETTER_Scripts/zeroshot/chatgpt/download_batch.py --language $lang --n $n --split $split --batch-id-file $BATCH_FILE --think --out-dir $OUT_DIR

