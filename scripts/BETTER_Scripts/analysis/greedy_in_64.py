import json
import argparse
import glob
import os

def greed_in_64_percentage(instance_greedy, instances_64):
    counter = 0.0
    for instance in instances_64:
        if instance_greedy == instance:
            counter += 1.0
    
    return counter/64.0

def greed_in_64(instance_greedy, instances_64):
    if instance_greedy in instances_64:
        return 1.0
    return 0.0

def same_in_64(instances_64):
    counter = 0.0
    previous_templates = []
    for instance in instances_64:
        if instance not in previous_templates:
            previous_templates.append(instance)
    return (64.0-len(previous_templates))/64.0


parser = argparse.ArgumentParser(description='Arguments required to do the analysis')
parser.add_argument('--input-path', dest='input_file', type=str)

args = parser.parse_args()
input_file = args.input_file
greedy_files = glob.glob(os.path.join(input_file, "*_1.jsonl"))
for greedy_file in greedy_files:
    modelname = os.path.basename(greedy_file).replace("_1.jsonl","")
    sampling_64_file = greedy_file.replace("_1.jsonl", "_64.jsonl")
    with open(greedy_file, 'r') as f:
        greedy_lines = f.readlines()
    with open(sampling_64_file, 'r') as f:
        sampling_64_lines = f.readlines()
    total_greed_in_64 = 0.0
    total_greed_appearence_in_64 = 0.0
    total_same_in_64 = 0.0
    num_instances = len(greedy_lines)
    for i in range(num_instances):
        greedy_instance = json.loads(greedy_lines[i])["templates"]
        sampling_64_instance = json.loads(sampling_64_lines[i])["templates"]
        greed_in_64_score = greed_in_64_percentage(greedy_instance, sampling_64_instance)
        same_in_64_score = same_in_64(sampling_64_instance)
        total_greed_appearence_in_64 += greed_in_64(greedy_instance, sampling_64_instance)
        total_greed_in_64 += greed_in_64_score
        total_same_in_64 += same_in_64_score
    avg_greed_in_64 = total_greed_in_64 / num_instances
    avg_same_in_64 = total_same_in_64 / num_instances
    avg_greed_appearence_in_64 = total_greed_appearence_in_64 / num_instances
    print(f"Model: {modelname}, Greedy_in_64: {avg_greed_in_64}, same_in_64: {avg_same_in_64} greedy_appearence_in_64: {avg_greed_appearence_in_64}")


