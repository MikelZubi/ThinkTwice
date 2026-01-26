import sys
sys.path.append('BETTER_Scorer/')
sys.path.append("scripts/BETTER_Scripts/results")
from postprocess_BETTER import dict_postprocessed_data
import argparse
import numpy as np
import json
from tqdm import tqdm
import score
from lib.bp import BPDocument

def template_voting(line):
    #TODO: Oaintxe bertan errore bat scorer-ean, hypotesia: ezin da eventu utsa duen template-ak prozesatu, komprobatu eta ala bada gehitu gezurretazko eventu bat ("a" esaterako bietan)
    id = line["docid"]
    document = line["doctext"]
    templates = line["templates"]

    cleaned_templates = [template for template in templates if "ERROR" not in template or "ERROR" not in template[0]]
    mean_templates = []
    for out_template1 in cleaned_templates:
        current_scores = 0.0
        for out_template2 in cleaned_templates:
            if out_template1 == out_template2:
                current_scores += 1.0
        mean_score = current_scores / len(cleaned_templates)
        mean_templates.append(mean_score)
    print(mean_templates)
    if len(mean_templates) == 0:
        return ["ERROR"]
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
with open(read_file, 'r') as file:
    for line in tqdm(file):
        data = json.loads(line)
        pre_dict = {}
        pre_dict["docid"] = data["docid"]
        pre_dict["doctext"] = data["doctext"]
    
        max_template = template_voting(data)
        pre_dict["templates"] = max_template
        pre_dicts.append(pre_dict)

with open(out_dir, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')