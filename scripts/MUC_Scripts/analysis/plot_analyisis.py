import json
import csv
from matplotlib import pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Plot analysis results for MUC dataset')
parser.add_argument('--split', dest='split', type=str, default="dev")
args = parser.parse_args()
split = args.split
models = ["Qwen3-32B_think", "Qwen3-32B_nothink", "LlamaR1-70B", "Llama3.3-70B"]

greedy_results = {}
greedy_path = f"results/MUC/zeroshot/{split}/en.csv"

with open(greedy_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        greedy_results = {}
        voter_f1_results = {}
        for row in reader:
                model = row['file'].replace("_1.jsonl", "")
                if model in models:
                        greedy_results[model] = float(row['f1'])

voter_f1_results = {}
voter_f1_path = f"results/MUC/zeroshot/{split}/en/voterf1.csv"
with open(voter_f1_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                model = row['file'].replace("_64.jsonl", "")
                if model in models:
                        voter_f1_results[model] = float(row['f1'])


csv_path = f"results/MUC/zeroshot/{split}/en_analysis.csv"
model_stats = {}

with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
                model = row['modelname']
                if model in models:
                        mean = float(row['random_mean_score'])
                        std = float(row['random_std_score'])
                        max_val = float(row['max_score'])
                        model_stats[model] = {'mean': mean, 'std': std, 'max': max_val, "greedy": greedy_results[model], "voter_f1": voter_f1_results[model]}

means = [model_stats[m]['mean'] for m in models]
stds = [model_stats[m]['std'] for m in models]
maxs = [model_stats[m]['max'] for m in models]
greedys = [model_stats[m]['greedy'] for m in models]
voters = [model_stats[m]['voter_f1'] for m in models]

# Create figure
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(models))

# Plot error bars centered at mean with std
ax.errorbar(x, means, yerr=stds, fmt='o', color='#3498db', markersize=10, 
            capsize=8, capthick=2, elinewidth=2, label='Mean ± Std', 
            alpha=0.7, zorder=1)

# Plot greedy and voter_f1 as scatter points
ax.scatter(x, greedys, marker='s', s=120, color='#2ecc71', 
           label='Greedy', edgecolors='black', linewidth=1.5, zorder=3)
ax.scatter(x, voters, marker='^', s=120, color='#f39c12', 
           label='Voter F1', edgecolors='black', linewidth=1.5, zorder=3)

# Add value labels
for i in range(len(models)):
        ax.text(x[i], means[i] + 0.006, f'{means[i]:.3f}', ha='center', va='bottom', 
        fontsize=9, color='#3498db', fontweight='bold')
        ax.text(x[i] + 0.12, greedys[i], f'{greedys[i]:.3f}', ha='left', va='center', 
        fontsize=9, color='#2ecc71', fontweight='bold')
        ax.text(x[i] + 0.12, voters[i], f'{voters[i]:.3f}', ha='left', va='center', 
        fontsize=9, color='#f39c12', fontweight='bold')

# Styling
ax.set_ylabel('F1 Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Model', fontsize=13, fontweight='bold')
#ax.set_title('Model Performance Analysis on MUC Dataset', fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha='right', fontsize=11)

# Smaller axis range
y_min = min(means + voters + greedys) - 0.0125
y_max = max(means + voters + greedys) + 0.0125
ax.set_ylim(y_min, y_max)

# Legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, framealpha=0.95, edgecolor='black')
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Add max values as text box outside the plot
max_text = "Max Scores:\n" + "\n".join([f"{models[i]}: {maxs[i]:.3f}" for i in range(len(models))])
ax.text(1.02, 0.5, max_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f"irudiak/MUC_analisis_{split}.pdf", bbox_inches='tight')
plt.close()