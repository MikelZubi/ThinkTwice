import argparse
import json
import csv
import glob 
import os
import numpy as np
from max_score import max_score
from random_scores import random_scores

parser = argparse.ArgumentParser(description='Arguments required to create analysis')
parser.add_argument('--read', dest='read', type=str)
parser.add_argument("--split", dest="split", type=str, default='dev')

def read_csv_file(file_path, modelname):
    with open(file_path, newline='') as csvfile:
        for row in csv.DictReader(csvfile):
            if modelname in row['file']:
                return float(row["better"])

args = parser.parse_args()
read = args.read
gold_path = f'phase2/phase2.granular.eng.{args.split}.json'

prediction_files = glob.glob(os.path.join(read, "*_64.jsonl"))
output_file =  read + "_analysis.csv"
fieldnames = ['modelname', 'max_score', 'random_mean_score', 'random_std_score', 'greedy', 'voter_majority', 'voter_f1']

with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for file in prediction_files:
        modelname = os.path.basename(file).replace("_64.jsonl", "")
        greedy = read_csv_file(read+"_scores.csv", modelname)
        voter_majority = read_csv_file(read+"/votedMajority_scores.csv", modelname)
        voter_f1 = read_csv_file(read+"/voterf1_scores.csv", modelname)
        print(f"Processing model: {modelname}")
        max_sc = max_score(file, gold_path)
        random_sc_list = random_scores(file, gold_path, n=100)
        random_mean = np.mean(random_sc_list)
        random_std = np.std(random_sc_list)
        row = {
            'modelname': modelname,
            'max_score': max_sc * 100,
            'random_mean_score': random_mean * 100,
            'random_std_score': random_std * 100,
            'greedy': greedy * 100,
            'voter_majority': voter_majority * 100,
            'voter_f1': voter_f1 * 100
        }
        writer.writerow(row)
