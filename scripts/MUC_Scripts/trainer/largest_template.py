import json
import os
import argparse
from collections import defaultdict





def template_selection(templates):
    templates = list(filter((["ERROR"]).__ne__, templates))
    templates = list(filter(([["ERROR"]]).__ne__, templates))
    
    max_templates = -1
    max_entities = 0
    selected_template = []
    for template in templates:
        if len(template) > max_templates:
            current_entities = 0
            for temp in template:
                for key in temp.keys():
                    if key != "incident_type":
                        current_entities += len(temp[key])
            if current_entities > max_entities:
                max_templates = len(template)
                max_entities = current_entities
                selected_template = template
    return selected_template

    '''
    best_template = []
    for _ in range(maximum_num):
        slot_counter = {"incidet_type": defaultdict(int), "PerpInd": defaultdict(int), "PerpOrg": defaultdict(int), "Target": defaultdict(int), "Victim": defaultdict(int), "Weapon": defaultdict(int)}
        slot_num = {"PerpInd": defaultdict(int), "PerpOrg": defaultdict(int), "Target": defaultdict(int), "Victim": defaultdict(int), "Weapon": defaultdict(int)}
        for template in templates:
            if len(template) != maximum_num:
                continue
            for key in template.keys():
                if key == "incident_type":
                    slot_counter[key][template[key]] += 1
                else:
                    slot_num[key][len(template[key])] += 1
                    for element in template[key]:
                        slot_counter[key][element] += 1
        voted_template = {}
        for key in slot_counter.keys():
            if key == "incident_type":
                voted_template[key] = max(slot_counter[key], key=slot_counter[key].get)
            else:
    '''
    
#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--read', dest='read', type=str)
parser.add_argument("--out", dest="out", type=str)

args = parser.parse_args()
read_file = args.read
out_dir = args.out

max_templates = []
pre_dicts = []
with open(read_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        max_template = template_selection(data["pred_json"])
        pre_dict["pred_json_maxlen"] = max_template
        pre_dicts.append(pre_dict)


print("Done")
with open(out_dir, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



