import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import argparse

import sys
sys.path.append("class_data")
sys.path.append("inference_library")
from prompt_factory import prompt_factory
from copy import deepcopy
from utils import maxCommStr
from tqdm import tqdm



#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--predict-dir', dest='predict_dir', type=str)
parser.add_argument('--language', dest='language', type=str)
parser.add_argument("--model-dir", dest='model_dir', type=str)
parser.add_argument("--out-dir", dest="out_dir", type=str)

parser.set_defaults(model_dir="/scratch/ehu_p518_1/ehu_p518_1_1/Ereduak/DeepSeek-R1-Distill-Llama-70B")
parser.set_defaults(language="en")

args = parser.parse_args()
language = args.language
predict_dir = args.predict_dir
model_dir = args.model_dir
path_write = args.out_dir




LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}
language_name = LANGUAGE_MAP[language]

prompt_class = prompt_factory(model_dir, language_name, "MUC", think=False)



map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}

#path_read = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
if os.path.exists(path_write):
    os.remove(path_write)

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#model_name = "/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B"
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_dir) 
model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1,device_map="cuda")
inputs = []
pre_dicts = []



#STEP 1: Created the reasoning
inputs_all = []
templates_all = []
score_dict_all = []
pre_dict_all = []
with open(predict_dir, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        templates = [template for template in pre_dict["pred_json"] if template != ['ERROR'] and template != [['ERROR']]]
        templates_list = []
        for template in templates:
            if template not in templates_list:
                templates_list.append(template)
        inputs = []
        score_dict = {}
        for template in templates_list:
            str_template = json.dumps(template, ensure_ascii=False)
            score_dict[str_template] = 0.0
            prompt = prompt_class.generate_prompt(pre_dict, template=str_template)
            inputs.append(prompt)
        templates_all.append(templates_list)
        inputs_all.append(inputs)
        pre_dict_all.append(pre_dict)
        score_dict_all.append(score_dict)

max_last = 0
new_pred_dict_all = []
with torch.no_grad():
    for templates, inputs, pre_dict, score_dict in tqdm(zip(templates_all, inputs_all, pre_dict_all, score_dict_all), total=len(inputs_all)):
        best_template = []
        max_log = float("-inf")
        for inp, temp in zip(inputs, templates):
            logits = model(inp).logits
            logits_item = logits.item()
            str_template = json.dumps(temp, ensure_ascii=False)
            score_dict[str_template] = logits_item
            if logits_item > max_log:
                max_log = logits_item
                best_template = temp
        new_pre_dict = {
            "docid": pre_dict["docid"],
            "doctext": pre_dict["doctext"],
            "templates": pre_dict["templates"],
            "pred_json": best_template,
            "score_dict": score_dict
        }
        new_pred_dict_all.append(new_pre_dict)

# Write the results to a jsonl file
with open(path_write, 'w', encoding='utf-8') as f:
    for new_pre_dict in new_pred_dict_all:
        json_line = json.dumps(new_pre_dict, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Results written to {path_write}")




