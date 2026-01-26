#!/bin/bash
rm -rf results/MUC/zeroshot/test/boostrap_en/voterf1
rm -rf results/MUC/zeroshot/test/boostrap_en/voterMajority
rm results/MUC/zeroshot/test/boostrap_en/voterf1.csv
rm results/MUC/zeroshot/test/boostrap_en/voterMajority.csv
for file in results/MUC/zeroshot/test/boostrap_en/*; do
if [[ "$(basename "$file")" == *_1.jsonl ]]; then
    continue
fi
if [[ "$(basename "$file")" == "voterf1" || "$(basename "$file")" == "voterMajority" ]]; then
    continue
fi
    echo "Processing file: $file"
    outfile="results/MUC/zeroshot/test/boostrap_en/voterf1/$(basename "$file")"
    mkdir -p results/MUC/zeroshot/test/boostrap_en/voterf1
    python scripts/MUC_Scripts/zeroshot/scorer_voterf1.py --read "$file" --out "$outfile"
    outfile="results/MUC/zeroshot/test/boostrap_en/voterMajority/$(basename "$file")"
    mkdir -p results/MUC/zeroshot/test/boostrap_en/voterMajority
    python scripts/MUC_Scripts/zeroshot/scorer_voterMajority.py --read "$file" --out "$outfile"

done

python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/test/boostrap_en/voterf1 --split test --language en --voterf1
python scripts/MUC_Scripts/zeroshot/calculate_results.py --read results/MUC/zeroshot/test/boostrap_en/voterMajority --split test --language en --voter