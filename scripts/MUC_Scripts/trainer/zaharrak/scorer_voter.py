import json
import os
import argparse
from collections import defaultdict




def simplify_template(templates):
    new_templates = []
    for template in templates:
        new_template = {}
        for key in template.keys():
            if key=="incident_type":
                new_template[key] = template[key]
            elif template[key]==[]:
                new_template[key] = template[key]
            else:
                new_template[key] = []
                for element in template[key]:
                    new_template[key].append(element[0].lower())
        new_templates.append(new_template)
    dict_templates = {"templates": new_templates}
    return json.dumps(dict_templates, ensure_ascii=False)

def template_voting(templates):
    templates = list(filter((["ERROR"]).__ne__, templates))
    templates = list(filter(([["ERROR"]]).__ne__, templates))
    
    num_templates = defaultdict(int)
    for template in templates:
        num_templates[len(template)] += 1
    maximum = 0
    maximum_num = 0
    for num_template in num_templates:
        if num_templates[num_template] > maximum:
            maximum = num_templates[num_template]
            maximum_num = num_template
    if maximum_num == 0:
        return {"templates": []}
    template_counter = defaultdict(int)
    for template in templates:
        if len(template) != maximum_num:
            continue
        simp_template = simplify_template(template)
        template_counter[simp_template] += 1
    max_template = 0
    for template in template_counter:
        if template_counter[template] > max_template:
            max_template = template_counter[template]
            voted_template = template
    voted_template = json.loads(voted_template)
    if "templates" not in voted_template:
        print("Error: templates not in voted_template")
        print(voted_template)
    return voted_template


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
        max_template = template_voting(data["pred_json"])
        max_templates.append(max_template)
        pre_dicts.append(pre_dict)


for idx, outputs in enumerate(max_templates):
    post_templates = []
    for template in outputs["templates"]:
        post_processed = {}
        for key in template.keys():
            if key != "incident_type" and template[key] != []:
                post_processed[key]=[[elem.lower()] for elem in template[key]]
            else:
                post_processed[key]=template[key]
        post_templates.append(post_processed)
    pre_dicts[idx]["pred_json_scorer"] = post_templates

print("Done")
with open(out_dir, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



