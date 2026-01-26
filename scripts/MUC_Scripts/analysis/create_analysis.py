import argparse
import json
import csv
import glob 
import os
import numpy as np
from max_score import max_score
from random_scores import random_scores

parser = argparse.ArgumentParser(description='Arguments required to create analysis')
parser.add_argument("--split", dest="split", type=str, default='dev')

def read_csv_file(file_path, modelname):
    if "Reward" in file_path:
        find_modelname = modelname.replace("-32B","").replace("-70B","")
    else:
        find_modelname = modelname
    with open(file_path, newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            if find_modelname in row['file']:
                return float(row["f1"])
        print(f"{find_modelname} Not found. in {file_path}")
        return 0.0


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
        voter_f1 = read_csv_file(read+"/voterf1.csv", modelname)
        reward = read_csv_file(read+"/Reward.csv", modelname)
        print("Reward Score:", reward)
        row = {
            'language': language,
            'modelname': modelname,
            'max_score': max_sc * 100,
            'random_mean_score': random_mean * 100,
            'random_std_score': random_std * 100,
            'greedy': greedy * 100,
            'voter_majority': voter_majority * 100,
            'voter_f1': voter_f1 * 100,
            'reward': reward* 100
        }
        rows.append(row)

fieldnames = ['language', 'modelname', 'max_score', 'random_mean_score', 'random_std_score', 'greedy', 'voter_majority', 'voter_f1', 'reward']
#fieldnames = ['language', 'modelname', 'max_score', 'random_mean_score', 'random_std_score', 'greedy', 'voter_majority', 'voter_f1']
output_file =  "results/MUC/zeroshot/" + split + "/multilingual_analysis.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)