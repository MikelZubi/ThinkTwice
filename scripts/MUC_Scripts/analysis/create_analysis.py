import argparse
import json
import csv
import glob 
import os
import numpy as np
from max_score import max_score
from random_scores import random_scores

parser = argparse.ArgumentParser(description='Arguments required to create analysis')
parser.add_argument("--split", dest="split", type=str, default='test')


def read_csv_file(file_path, modelname,mean_std=False):
    with open(file_path, newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            if modelname in row['file']:
                if mean_std:
                    mean = float(row['f1'].split('±')[0]) * 100
                    std = float(row['f1'].split('±')[1]) * 100
                    output = str(mean) + "±" + str(std)
                else:
                    output = float(row['f1']) * 100
                return str(output)
        return "0.0"


languages = ["en", "ar", "fa", "ko", "ru", "zh"]
args = parser.parse_args()
split = args.split
rows = []
for language in languages:
    read = f"results/MUC/zeroshot/{split}/{language}"
    gold_path = f"multimuc/data/multimuc_v1.0/corrected/{language}/{split}.jsonl"
    prediction_files = glob.glob(os.path.join(read, "*_64.jsonl"))
    voterf1_path = os.path.join(read, "voterf1.csv")
    voterf1_scores = {}
    output_file =  read + "_analysis.csv"
    for file in prediction_files:
        modelname = os.path.basename(file).replace("_64.jsonl", "")
        print(f"Processing model: {modelname}")
        print("Language:", language)
        max_sc = max_score(file, gold_path)
        print("Max score:", max_sc)
        random_sc_list = random_scores(file, gold_path, n=100)
        random_mean = np.mean(random_sc_list)
        random_std = np.std(random_sc_list)
        greedy = read_csv_file(read+".csv", modelname)
        voter_majority = read_csv_file(read+"/voterMajority.csv", modelname)
        voter_majority_mean_std = read_csv_file(read+"/votermajority_mean_std.csv", modelname, mean_std=True)
        voter_f1 = read_csv_file(read+"/voterf1.csv", modelname)
        voter_f1_mean_std = read_csv_file(read+"/voterf1_mean_std.csv", modelname, mean_std=True)
        reward = read_csv_file(read+"/Reward.csv", modelname)
        reward_mean_std = read_csv_file(read+"/reward_mean_std.csv", modelname, mean_std=True)
        print("Reward Score:", reward)
        row = {
            'language': language,
            'modelname': modelname,
            'max_score': max_sc * 100,
            'random_mean_score': random_mean * 100,
            'random_std_score': random_std * 100,
            'greedy': greedy,
            'voter_majority': voter_majority,
            'voter_f1': voter_f1,
            'reward': reward,
            'voter_majority_mean_std': voter_majority_mean_std,
            'voter_f1_mean_std': voter_f1_mean_std,
            'reward_mean_std': reward_mean_std
        }
        rows.append(row)

fieldnames = ['language', 'modelname', 'max_score', 'random_mean_score', 'random_std_score', 'greedy', 'voter_majority', 'voter_f1', 'reward', 'voter_majority_mean_std', 'voter_f1_mean_std', 'reward_mean_std']
#fieldnames = ['language', 'modelname', 'max_score', 'random_mean_score', 'random_std_score', 'greedy', 'voter_majority', 'voter_f1']
output_file =  "results/MUC/zeroshot/" + split + "/multilingual_analysis.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)