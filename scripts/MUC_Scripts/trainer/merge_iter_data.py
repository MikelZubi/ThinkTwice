DATA_TO_MERGE = {0:"train_Reasoning_64.jsonl", 
                 1:"Sampling_8-checkpoint-447-NoGD_64.jsonl",
                 2:"Sampling_8-checkpoint-164-NoGD_64.jsonl",
                 3:"Sampling_8-checkpoint-164-NoGD_64.jsonl", 
                 4:"Sampling_8-checkpoint-82-NoGD_64.jsonl", 
                 5:"Sampling_8-checkpoint-164-NoGD_64.jsonl", 
                 6:"Sampling_8-checkpoint-246-NoGD_64.jsonl",
                 7:"Sampling_8-checkpoint-246-NoGD_64.jsonl",
                 8:"Sampling_8-checkpoint-82-NoGD_64.jsonl"}

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
parser.set_defaults(random_length=100)
args = parser.parse_args()
random_length = args.random_length
rd.seed(42)



merged_data = []


for key in tqdm(DATA_TO_MERGE):
    path = "rejectionSampling/train/"+str(key)+"/"+DATA_TO_MERGE[key]
    if merged_data == []:
        for line in open(path, 'r'):
            data = json.loads(line)
            merged_data.append(data)
    else:
        for i, line in enumerate(open(path, 'r')):
            data = json.loads(line)
            for template, reasoning in zip(data["pred_json"], data["pred_reasoning"]):
                if template == ["ERROR"] or template == [["ERROR"]]:
                    continue
                template_lower = template_to_lower(template)
                if template_lower not in merged_data[i]["pred_json"]:
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

output_path = "/leonardo/pub/userexternal/mzubilla/selected_merge_iter_data.jsonl"
with open(output_path, 'w') as outfile:
    for entry in selected_merged_data:
        json.dump(entry, outfile)
        outfile.write('\n')

if unselected_merged_data != []:
    output_path = "/leonardo/pub/userexternal/mzubilla/unselected_merge_iter_data.jsonl"
    with open(output_path, 'w') as outfile:
        for entry in unselected_merged_data:
            json.dump(entry, outfile)
            outfile.write('\n')
    