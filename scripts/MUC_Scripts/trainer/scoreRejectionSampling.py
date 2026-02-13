import json
import os
import numpy as np
import csv
import argparse
import sys

sys.path.append("scripts/MUC_Scripts/analysis")
from max_score import max_score
from random_scores import random_scores





parser = argparse.ArgumentParser(description='Arguments required for the scorer')
parser.add_argument('--split', dest='split', type=str)
parser.add_argument("--iter", dest='iter', type=str)
parser.add_argument("--noReasoning", dest='no_reasoning', action='store_true',)
parser.add_argument("--modelname", dest='modelname', type=str)
parser.add_argument("--starting-path", dest='starting_path', type=str)
parser.set_defaults(split="dev")
parser.set_defaults(iter=1)
parser.set_defaults(no_reasoning=False)
parser.set_defaults(modelname="QWEN")
parser.set_defaults(starting_path="")
args = parser.parse_args()
split = args.split
iteration = args.iter
modelname = args.modelname
starting_path = args.starting_path


#READ GOLD
gold_path = "multimuc/data/multimuc_v1.0/corrected/en/"+split+".jsonl"#IDATZI

completions = []
#paths = {"Reasoning": lambda x: "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/dev_Reasoning_"+str(x)+".jsonl", "StepReasoning": lambda x: "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/dev_StepReasoning_"+str(x)+".jsonl", "JSON": lambda x: "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/dev_JSON_"+str(x)+".jsonl"}

paths = {}
#Iterate files in rejectionSampling/dev
#for file in os.listdir("rejectionSampling/"+split):
tag = starting_path + "/" + ( "rejectionSampling/"+modelname + "/" if not args.no_reasoning else "NoReasoning/"+modelname + "/")
for file in os.listdir(tag+split+"/"+iteration+"/"):
    if file.endswith("_64.jsonl"):
        #Extract the type and n from the filename
        file_type = "_".join(file.split("_")[:-1])
        #Add the path to the dictionary
        paths[file_type] = lambda x, ct=file_type: tag+split+"/"+iteration+"/" + ct + "_" + str(x) + ".jsonl"
        #paths[file_type] = lambda x, ct=file_type: "rejectionSampling/"+split+"/" + ct + "_" + str(x) + ".jsonl"



header = ["Type","n","MAX","STD","Mean"]
out_list = []
ns = [64]
stds = []
means = []
for key in paths:
    for n in ns:
        stds.clear()
        path = paths[key](n)
        best_f1s = []
        dis = 0
        best_templates = {}
        print("Processing:",path)
        max_sc = max_score(path, gold_path) * 100
        random_sc_list = random_scores(path, gold_path, n=100)
        random_mean = np.mean(random_sc_list) * 100
        random_std = np.std(random_sc_list) * 100 
        out_list.append([key,n,max_sc,random_std,random_mean])

out_path = tag+split+"/scores_iter"+str(iteration)+".csv"
#out_path = "rejectionSampling/"+split+"/scores.csv"
with open(out_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(out_list)