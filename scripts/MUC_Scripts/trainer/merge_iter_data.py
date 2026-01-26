DATA_TO_MERGE_LLAMA = {0:"train_Reasoning_64.jsonl",
                 1:"Sampling_8-checkpoint-447-NoGD_64.jsonl",
                 2:"Sampling_8-checkpoint-164-NoGD_64.jsonl",
                 3:"Sampling_8-checkpoint-164-NoGD_64.jsonl", 
                 4:"Sampling_8-checkpoint-82-NoGD_64.jsonl", 
                 5:"Sampling_8-checkpoint-164-NoGD_64.jsonl", 
                 6:"Sampling_8-checkpoint-246-NoGD_64.jsonl",
                 7:"Sampling_8-checkpoint-246-NoGD_64.jsonl",
                 8:"Sampling_8-checkpoint-82-NoGD_64.jsonl"}

DATA_TO_MERGE_QWEN = {0:"Reasoning_64.jsonl",
1: "Sampling_8-checkpoint-328-NoGD_64.jsonl",
2: "Sampling_8-checkpoint-164-NoGD_64.jsonl",
3: "Sampling_8-checkpoint-82-NoGD_64.jsonl",
4: "Sampling_8-checkpoint-246-NoGD_64.jsonl",
5: "Sampling_8-checkpoint-246-NoGD_64.jsonl",
6: "Sampling_8-checkpoint-164-NoGD_64.jsonl",
7: "Sampling_8-checkpoint-164-NoGD_64.jsonl",
8: "Sampling_8-checkpoint-164-NoGD_64.jsonl",
9: "Sampling_8-checkpoint-82-NoGD_64.jsonl",
10: "Sampling_8-checkpoint-164-NoGD_64.jsonl",
11: "Sampling_8-checkpoint-82-NoGD_64.jsonl",
12: "Sampling_8-checkpoint-82-NoGD_64.jsonl",
13: "Sampling_8-checkpoint-164-NoGD_64.jsonl",
14: "Sampling_8-checkpoint-164-NoGD_64.jsonl"}

import json
from tqdm import tqdm
import random as rd
import copy as cp
import argparse


def template_to_lower(templates):
    new_templates = []
    for template in templates:
        new_template = cp.deepcopy(template)
        for key in template:
            if key != "incident_type":
                for i in range(len(template[key])):
                    new_template[key][i][0] = template[key][i][0].lower()
        new_templates.append(new_template)
    return new_templates

#TODO: Random-ekin template-ak aukeratu biño lehenik errepikatutako template-ak kendu egin behar dira
#Horretarako funtzio bat in beharkoa bi template komparatu eta berdiñak diren a la ez itzuliko duena
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--random-length', dest='random_length', type=int)
parser.add_argument("--modelname", type=str, choices=['LLAMA', 'QWEN'], help="Model name to determine which data to merge")
parser.add_argument("--same-template", dest="same_template", action='store_true', help="If set, only keep one instance of each template")
parser.set_defaults(same_template=False)
parser.set_defaults(random_length=100)
parser.set_defaults(modelname='QWEN')
args = parser.parse_args()
random_length = args.random_length
modelname = args.modelname
same_template = args.same_template

if modelname == 'LLAMA':
    data_to_merge = DATA_TO_MERGE_LLAMA
else:
    data_to_merge = DATA_TO_MERGE_QWEN
rd.seed(42)



merged_data = []


for key in tqdm(data_to_merge):
    path = f"rejectionSampling/{modelname}/train/{key}/{data_to_merge[key]}"
    if merged_data == []:
        for line in open(path, 'r'):
            data = json.loads(line)
            merged_data.append(data)
    else:
        for i, line in enumerate(open(path, 'r')):
            data = json.loads(line)
            for template, reasoning in zip(data["pred_json"], data["pred_reasoning"]):
                if "ERROR" in template or ["ERROR"] in template:
                    continue
                template_lower = template_to_lower(template)

                if template_lower not in merged_data[i]["pred_json"] or (reasoning not in merged_data[i]["pred_reasoning"] and not same_template):
                    merged_data[i]["pred_json"].append(template_lower)
                    merged_data[i]["pred_reasoning"].append(reasoning)
            assert merged_data[i]["templates"] == data["templates"]
            assert merged_data[i]["docid"] == data["docid"]

selected_merged_data = []
unselected_merged_data = []
for i in range(len(merged_data)):
    selected_merged_data.append({})
    unselected_merged_data.append({})
    if len(merged_data[i]["pred_json"]) > random_length:
        sampled_indices = rd.sample(range(len(merged_data[i]["pred_json"])), random_length)
        unselected_indices = [j for j in range(len(merged_data[i]["pred_json"])) if j not in sampled_indices]
        selected_merged_data[i]["pred_json"] = [merged_data[i]["pred_json"][j] for j in sampled_indices]
        selected_merged_data[i]["pred_reasoning"] = [merged_data[i]["pred_reasoning"][j] for j in sampled_indices]
        unselected_merged_data[i]["pred_json"] = [merged_data[i]["pred_json"][j] for j in unselected_indices]
        unselected_merged_data[i]["pred_reasoning"] = [merged_data[i]["pred_reasoning"][j] for j in unselected_indices]
    else:
        selected_merged_data[i] = merged_data[i]
        unselected_merged_data[i] = {}

output_path = "/home/ehu_p518_1/COMUNES/mikel/selected_merge_iter_data.jsonl"
with open(output_path, 'w') as outfile:
    for entry in selected_merged_data:
        json.dump(entry, outfile)
        outfile.write('\n')

if unselected_merged_data != []:
    output_path = "/home/ehu_p518_1/COMUNES/mikel/unselected_merge_iter_data.jsonl"
    with open(output_path, 'w') as outfile:
        for entry in unselected_merged_data:
            json.dump(entry, outfile)
            outfile.write('\n')
    