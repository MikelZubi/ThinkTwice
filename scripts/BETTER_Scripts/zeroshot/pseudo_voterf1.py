import sys
import argparse
import numpy as np
import json
from tqdm import tqdm
from pseudo_scorer import make_triplets, evaluate_triples

def template_voting(line):
    #TODO: Oaintxe bertan errore bat scorer-ean, hypotesia: ezin da eventu utsa duen template-ak prozesatu, komprobatu eta ala bada gehitu gezurretazko eventu bat ("a" esaterako bietan)
    id = line["docid"]
    document = line["doctext"]
    templates = line["pred_json"]

    cleaned_templates = [template for template in templates if "ERROR" not in template or "ERROR" not in template[0]]
    if cleaned_templates == []:
        return ["ERROR"]
    mean_templates = []
    for template_data1 in cleaned_templates:
        current_scores = 0.0
        for template_data2 in cleaned_templates:
            if template_data1 == template_data2:
                current_scores += 1.0
                continue

            triplet_template1 = make_triplets(template_data1)
            triplet_template2 = make_triplets(template_data2)
            current_score = evaluate_triples(triplet_template2, triplet_template1)
            current_scores += current_score
        mean_score = current_scores / len(cleaned_templates)
        mean_templates.append(mean_score)

    maximum_num = np.argmax(mean_templates)
    best_template = cleaned_templates[maximum_num]
    return best_template


#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--read', dest='read', type=str)
parser.add_argument("--out", dest="out", type=str)

args = parser.parse_args()
read_file = args.read
out_dir = args.out

max_templates = []
pre_dicts = []
error_count = 0
with open(read_file, 'r') as file:
    for line in tqdm(file):
        data = json.loads(line)
        pre_dict = {}
        pre_dict["docid"] = data["docid"]
        pre_dict["doctext"] = data["doctext"]
    
        max_template = template_voting(data)
        if max_template == ["ERROR"]:
            error_count += 1
        pre_dict["pred_json"] = max_template
        pre_dicts.append(pre_dict)
print("\n\n\n********************TOTAL ERRORS********************\n")
print(error_count)

with open(out_dir, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')