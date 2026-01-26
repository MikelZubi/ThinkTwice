#!/usr/bin/env python3
"""
Script to create summary tables from scorer_voterf1.slurm results.
Permite combinar métricas y conteos de errores para greedy, voterf1, majority y reward.
"""
import argparse
import pandas as pd
from pathlib import Path

# Define languages and models
LANGUAGES = ["en", "ar", "fa", "ko", "ru", "zh"]
MODELS = [
    "Qwen3_think",
    "Qwen3_nothink",
    "LlamaR1",
    "Llama3.3",
    #"MUCR1",
    #"MUCLLAMA",
]

DATASETS = {
    "greedy": {
        "description": "Greedy",
        "path_template": "results/MUC/zeroshot/{split}/{lang}.csv",
        "column_suffix": "greedy",
        "error_output": "error_counts_table.csv",
    },
    "voterf1": {
        "description": "VoterF1",
        "path_template": "results/MUC/zeroshot/{split}/{lang}/voterf1.csv",
        "column_suffix": "voterf1",
        "error_output": "error_counts_voterf1_table.csv",
    },
    "majority": {
        "description": "Majority Voting",
        "path_template": "results/MUC/zeroshot/{split}/{lang}/voterMajority.csv",
        "column_suffix": "majority",
        "error_output": "error_counts_majority_table.csv",
    },
    "reward": {
        "description": "Reward",
        "path_template": "results/MUC/zeroshot/{split}/{lang}/Reward.csv",
        "column_suffix": "reward",
        "error_output": "error_counts_reward_table.csv",
    },
}

DEFAULT_DATASET_ORDER = ["greedy", "voterf1", "majority", "reward"]

MODEL_ALIASES = {
    "Qwen3_think": ["Qwen3_think", "Qwen3-32B_think"],
    "Qwen3_nothink": ["Qwen3_nothink", "Qwen3-32B_nothink"],
    "Llama3.3": ["Llama3.3", "Llama3.3-70B"],
    "LlamaR1": ["LlamaR1", "LlamaR1-70B"],
    "MUCR1": ["MUCR1"],
    "MUCLLAMA": ["MUCLLAMA"],
}

def load_csv_data(csv_path: str):
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    return None

def get_model_row(df: pd.DataFrame | None, model: str):
    if df is None or "file" not in df.columns:
        return None
    aliases = MODEL_ALIASES.get(model, [model])
    file_series = df["file"].astype(str)
    for alias in aliases:
        mask = file_series.str.contains(alias, na=False, regex=False)
        if mask.any():
            return df.loc[mask].iloc[0]
    return None

def create_f1_comparison_table(split: str, dataset_keys: list[str]) -> pd.DataFrame:
    cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    rows = []
    for model in MODELS:
        row = {"Model": model}
        for lang in LANGUAGES:
            for dataset_key in dataset_keys:
                cache_key = (dataset_key, lang)
                if cache_key not in cache:
                    path = DATASETS[dataset_key]["path_template"].format(split=split, lang=lang)
                    cache[cache_key] = load_csv_data(path)
                df = cache[cache_key]
                model_row = get_model_row(df, model)
                value = None
                if model_row is not None and "f1" in model_row.index and not pd.isna(model_row["f1"]):
                    value = model_row["f1"]
                column_name = f"{lang}_{DATASETS[dataset_key]['column_suffix']}"
                row[column_name] = f"{float(value):.4f}" if value is not None else "N/A"
        rows.append(row)
    df = pd.DataFrame(rows)
    ordered_columns = ["Model"]
    for lang in LANGUAGES:
        for dataset_key in dataset_keys:
            ordered_columns.append(f"{lang}_{DATASETS[dataset_key]['column_suffix']}")
    return df[ordered_columns]

def create_error_count_table(split: str, dataset_key: str) -> pd.DataFrame:
    cache: dict[str, pd.DataFrame | None] = {}
    rows = []
    for model in MODELS:
        row = {"Model": model}
        for lang in LANGUAGES:
            if lang not in cache:
                path = DATASETS[dataset_key]["path_template"].format(split=split, lang=lang)
                cache[lang] = load_csv_data(path)
            df = cache[lang]
            model_row = get_model_row(df, model)
            value = None
            if model_row is not None and "num_errors" in model_row.index and not pd.isna(model_row["num_errors"]):
                value = model_row["num_errors"]
            if value is None:
                row[lang] = "N/A"
            else:
                try:
                    row[lang] = int(float(value))
                except (TypeError, ValueError):
                    row[lang] = value
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(description="Generar tablas resumen para resultados MUC zeroshot.")
    parser.add_argument("--split", type=str, default="dev", help="Split a usar (p. ej. dev, test).")
    parser.add_argument(
        "--include",
        nargs="+",
        choices=list(DATASETS.keys()) + ["all"],
        default=["all"],
        help="Tipos de resultados a incluir en las tablas (por defecto todos).",
    )
    args = parser.parse_args()
    split = args.split
    include_args = args.include
    if "all" in include_args:
        dataset_keys = DEFAULT_DATASET_ORDER
    else:
        dataset_keys = []
        for key in include_args:
            if key in DATASETS and key not in dataset_keys:
                dataset_keys.append(key)
        if not dataset_keys:
            dataset_keys = DEFAULT_DATASET_ORDER
    output_dir = Path(f"results/MUC/zeroshot/{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_descriptions = [DATASETS[key]["description"] for key in dataset_keys]
    print("Creating summary tables from scorer_voterf1.slurm results...\n")
    print("=" * 80)
    print(f"Table 1: F1 Scores Comparison ({', '.join(selected_descriptions)})")
    print("=" * 80)
    f1_table = create_f1_comparison_table(split, dataset_keys)
    print(f1_table.to_string(index=False))
    print("\n")
    f1_output = output_dir / "f1_comparison_table.csv"
    f1_table.to_csv(f1_output, index=False)
    print(f"Table saved to: {f1_output}\n")
    for idx, dataset_key in enumerate(dataset_keys, start=2):
        description = DATASETS[dataset_key]["description"]
        print("=" * 80)
        print(f"Table {idx}: Error Counts ({description} Predictions)")
        print("=" * 80)
        error_table = create_error_count_table(split, dataset_key)
        print(error_table.to_string(index=False))
        print("\n")
        error_output = output_dir / DATASETS[dataset_key]["error_output"]
        error_table.to_csv(error_output, index=False)
        print(f"Table saved to: {error_output}\n")
    print("All tables created successfully!")

if __name__ == "__main__":
    main()
