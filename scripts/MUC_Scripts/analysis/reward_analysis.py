import json
from scipy.stats import zscore, spearmanr
from numpy import arctanh as fisher
from numpy import tanh as inverse_fisher
import numpy as np
import argparse
from tqdm import tqdm
import csv



def calculate_total_correlation(file_path):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    total_pred_correlations = np.array([])
    total_real_correlations = np.array([])
    for entry in data:
        pred_scores = entry["predict_scores"]
        real_scores = entry["real_scores"]
        if len(set(pred_scores)) > 1 and len(set(real_scores)) > 1:
            pred_scores_z = zscore(pred_scores)
            total_pred_correlations = np.concatenate((total_pred_correlations, pred_scores_z),axis=None)
            total_real_correlations = np.concatenate((total_real_correlations, real_scores),axis=None)
    overall_correlation, overall_p_value = spearmanr(total_pred_correlations, total_real_correlations)
    return overall_correlation, overall_p_value


def calculate_mean_correlation(file_path, min_p_value=0.01):
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    correlations = []
    for entry in data:
        correlation = entry["correlation"]
        p_value = entry["p_value"]
        if p_value <= min_p_value:
            if correlation >= 1.0:
                correlation = correlation - 1e-5
            elif correlation <= -1.0:
                correlation = correlation + 1e-5
            correlations.append(correlation)
    np_correlations = np.array(correlations)
    correlaions_fisher = fisher(np_correlations)
    mean_correlation_fisher = inverse_fisher(np.mean(correlaions_fisher))
    mean_correlation_nofisher = np.mean(np_correlations)
    return mean_correlation_fisher, mean_correlation_nofisher



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze reward correlations")
    parser.add_argument("--language", type=str, required=True, help="Language code for analysis")
    parser.add_argument("--min-p-value", dest="min_p_value", type=float, default=0.01, help="Minimum p-value threshold for mean correlation calculation")
    parser.add_argument("--split", type=str, default="test", help="Data split to analyze (e.g., 'test', 'train')")
    args = parser.parse_args()
    language = args.language
    min_p_value = args.min_p_value
    split = args.split
    models = ["MUCQWEN_think", "MUCQWEN_nothink", "Qwen3-32B_think", "Qwen3-32B_nothink", "LlamaR1-70B", "MUCR1", "Llama3.3-70B", "MUCLLAMA"]
    print("Calculating reward correlations...")
    rows = [["Model","Overall Correlation","Overall P-value","Mean Correlation Fisher","Mean Correlation No Fisher"]]
    for model in tqdm(models):
        file_path = f"results/MUC/zeroshot/analysis/{split}/{language}/{model}_reward_correlation.jsonl"
        overall_correlation, overall_p_value = calculate_total_correlation(file_path)
        mean_correlation_fisher, mean_correlation_nofisher = calculate_mean_correlation(file_path, min_p_value)
        rows.append([model, overall_correlation, overall_p_value, mean_correlation_fisher, mean_correlation_nofisher])
    with open(f"results/MUC/zeroshot/analysis/{split}/{language}/reward_correlation_summary.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Calculating reward correlations with repeated templates removed...")
    rows = [["Model","Overall Correlation","Overall P-value","Mean Correlation Fisher","Mean Correlation No Fisher"]]
    for model in tqdm(models):
        file_path = f"results/MUC/zeroshot/analysis/{split}/{language}/{model}_reward_correlation_remove_repeated.jsonl"
        overall_correlation, overall_p_value = calculate_total_correlation(file_path)
        mean_correlation_fisher, mean_correlation_nofisher = calculate_mean_correlation(file_path, min_p_value)
        rows.append([model, overall_correlation, overall_p_value, mean_correlation_fisher, mean_correlation_nofisher])
    with open(f"results/MUC/zeroshot/analysis/{split}/{language}/reward_correlation_summary_remove_repeated.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    



