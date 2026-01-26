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
    out_templates = []
    post_templates = []
    for template in cleaned_templates:
        try:

            line_template = [{"docid": id, "doctext": document, "templates": template}] 
            post_process_template, error_count = dict_postprocessed_data(line_template, simplified=True, dict_input=True)
            template_data = BPDocument.from_dict(post_process_template)
            if not template_data.is_valid():
                continue
            post_templates.append(template_data)
            out_templates.append(template)
        except Exception as e:
            continue
    mean_templates = []
    for template_data1, out_template1 in zip(post_templates, out_templates):
        current_scores = 0.0
        for template_data2, out_template2 in zip(post_templates, out_templates):
            if out_template1 == out_template2:
                current_scores += 1.0
                continue

            try:
                #score_basic,_ = score.score_basic(template_data2, template_data1, no_validation=True, granular=True)
                score_granular, _ = score.score_granular(template_data2, template_data1, no_validation=True)
                current_score = score_granular.combined_score
                #current_score = score_granular.combined_score + score_basic.combined_score
                #text1 = json.dumps(out_template1)
                #print(text1)
                #text2 = json.dumps(out_template2)
                #print(text2)
                #num_events_1 = text1.count("event_type")
                #num_events_2 = text2.count("event_type")
                #print(f"NOT ERROR Number of events in template 1: {num_events_1}, Number of events in template 2: {num_events_2}")
                #score_basic, pairs = score.score_basic(template_data2, template_data1, no_validation=True, extras=True, granular=True)
                #current_score = score_granular.combined_score + score_basic.combined_score
            except Exception as e:
            #except json.JSONDecodeError as e:   
                #score_basic, _ = score.score_basic(template_data2, template_data1, no_validation=True, granular=True)
                #current_score = score_basic.combined_score
                current_score = 0.0
            '''
                print(e)
                current_score = 0.0
                text1 = json.dumps(out_template1)
                text2 = json.dumps(out_template2)
                print(text1)
                print(text2)
                num_events_1 = text1.count("event_type")
                num_events_2 = text2.count("event_type")
                print(f"ERROR Number of events in template 1: {num_events_1}, Number of events in template 2: {num_events_2}")
                exit()
            '''
            #print(current_score)
            current_scores += current_score
        mean_score = current_scores / len(post_templates)
        mean_templates.append(mean_score)
    print(mean_templates)
    if len(mean_templates) == 0:
        return ["ERROR"]
    maximum_num = np.argmax(mean_templates)
    best_template = out_templates[maximum_num]
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
        pre_dict["templates"] = max_template
        pre_dicts.append(pre_dict)
print("\n\n\n********************TOTAL ERRORS********************\n")
print(error_count)

with open(out_dir, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')