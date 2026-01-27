# ThinkTwice

This repository contains the code for the paper **"ThinkTwice"**, a sampling and selection framework for Document-level Information Extraction (DocIE): [https://arxiv.org/abs/2601.18395](https://arxiv.org/abs/2601.18395)

## 📊 Datasets

- **MUC/MultiMUC**: Terrorist incidents template filling ([MultiMUC](https://github.com/wgantt/multimuc))
- **BETTER**: Fine-grained event extraction ([BETTER Project](https://ir.nist.gov/better))

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 📁 Structure

Both datasets follow the same folder organization under `scripts/`:

| Folder | Description |
|--------|-------------|
| `preprocess/` | Data preprocessing scripts |
| `zeroshot/` | Zero-shot inference (`inference.py`) + selectors |
| `trainer/` | Training scripts (MUC only) |
| `results/` | Result processing |
| `analysis/` | Analysis and plots |

---

## 🚀 Usage

### 1. Preprocessing

```bash
# MUC
python scripts/MUC_Scripts/preprocess/preprocess_MUC_simplified.py

# BETTER
python scripts/BETTER_Scripts/preprocess/preprocess_BETTER_simplified.py
```

### 2. Zero-Shot Inference

Main inference script: `inference.py` (available in both `zeroshot/` folders)

```bash
python scripts/<DATASET>_Scripts/zeroshot/inference.py \
    --split train --n 32 --language en \
    --model-name <model-path> --out-dir <output-path> \
    --think  # Enable reasoning
```

**Selectors** (for selecting best outputs from multiple samples):
- `scorer_voterf1.py` / `pseudo_voterf1.py`
- `scorer_voterMajority.py` / `voterSimilarity.py`

### 3. Training (MUC Only)

| Script | Description |
|--------|-------------|
| `iteration.sh` | Iterative reasoning model training pipeline |
| `Reward.py` | Reward model training |
| `SFT.py` | Supervised Fine-Tuning |

```bash
# Start iterative training
bash scripts/MUC_Scripts/trainer/iteration.sh 1
```

### 4. Evaluation

Uses the IterX environment. Run `calculate_results.py` (available in both `zeroshot/` folders):

```bash
python scripts/<DATASET>_Scripts/zeroshot/calculate_results.py
```
