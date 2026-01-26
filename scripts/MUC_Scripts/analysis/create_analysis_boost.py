import csv
import glob 
import os
import numpy as np
from max_score import max_score
from random_scores import random_scores


def read_csv_file(file_path, modelname):
    with open(file_path, newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            if modelname in row['file']:
                return float(row["f1"])


sizes = [2,4,8,16,32,64]
language = "en"
split = "test"
read = f"results/MUC/zeroshot/{split}/boostrap_{language}"
rows = []
for size in sizes:
    gold_path = f"multimuc/data/multimuc_v1.0/corrected/{language}/{split}.jsonl"
    prediction_files = glob.glob(os.path.join(read, f"*_{size}.jsonl"))
    voterf1_path = os.path.join(read, "voterf1.csv")
    voterf1_scores = {}
    for file in prediction_files:
        modelname = os.path.basename(file).replace(".jsonl", "")
        real_modelname = modelname.replace(f"_{size}", "")
        print(f"Processing model: {modelname}")
        max_sc = max_score(file, gold_path)
        random_sc_list = random_scores(file, gold_path, n=100)
        random_mean = np.mean(random_sc_list)
        random_std = np.std(random_sc_list)
        random_vars = np.var(random_sc_list)
        greedy = read_csv_file(read+".csv", real_modelname)
        voter_majority = read_csv_file(read+"/voterMajority.csv", modelname)
        voter_f1 = read_csv_file(read+"/voterf1.csv", modelname)
        #reward = read_csv_file(read+"/Reward.csv", modelname)
        row = {
            "size": size,
            'modelname': real_modelname,
            'max_score': max_sc * 100,
            'random_mean_score': random_mean * 100,
            'random_std_score': random_std * 100,
            'random_var_score': random_vars * 100,
            'greedy': greedy * 100,
            'voter_majority': voter_majority * 100,
            'voter_f1': voter_f1 * 100,
            #'reward': reward* 100
        }
        rows.append(row)

#fieldnames = ['language', 'modelname', 'max_score', 'random_mean_score', 'random_std_score', 'greedy', 'voter_majority', 'voter_f1', 'reward']
fieldnames = ['size', 'modelname', 'max_score', 'random_mean_score', 'random_std_score', 'random_var_score', 'greedy', 'voter_majority', 'voter_f1']
output_file =  read + "_analysis.csv"
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)