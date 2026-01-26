import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import os
import argparse

import sys
sys.path.append("class_data")
sys.path.append("prompt_library")
from MUC_Class_simplified import *
from init import PROMPT_FN
from copy import deepcopy
from utils import maxCommStr
from tqdm import tqdm


def simplify_template(templates, gold_template=False):
    new_templates = []
    if not gold_template:
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
                        new_template[key].append(element[0])
            new_templates.append(new_template)
    else:
        new_templates = templates
    return json.dumps(new_templates, ensure_ascii=False)

def generate_prompt_encoder(data,tokenizer,template):
    prompt = "[CLS] \n" + data["doctext"] + "\n [SEP] \n" + template
    inputs = tokenizer(prompt).to("cuda")
    return inputs

def generate_prompt(data,tokenizer,language_code,template,guidelines=False,gold_template=False):
    language = LANGUAGE_MAP[language_code]
    if not guidelines:
        prompt = [{'role': 'system', 'content': PROMPT_FN["P_S_MUC_LLAMA_JSON_REWARD"].format(language=language)}]
        prompt.append({'role': 'user', 'content': PROMPT_FN["P_U_MUC_LLAMA_JSON_REWARD"].format(document=data["doctext"])})
    else:
        prompt = [{'role': 'system', 'content': PROMPT_FN["P_S_MUC_LLAMA_JSON"].format(language=language)}]
        prompt.append({"role": "user", "content": PROMPT_FN["P_U_MUC_LLAMA_JSON"].format(document=data["doctext"])})
    simp_template = simplify_template(template,gold_template=gold_template)
    prompt.append({"role": "assistant", "content": simp_template})
    prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=False, return_tensors="pt").to("cuda")
    return prompt_token_ids

#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--split', dest='split', type=str)
parser.add_argument('--predict', dest='predict', type=str)
parser.add_argument('--language', dest='language', type=str)
parser.add_argument("--model-name", dest='model_name', type=str)
parser.add_argument("--out-dir", dest="out_dir", type=str)
parser.add_argument("--guidelines", dest="guidelines", action="store_true")

parser.set_defaults(model_name="/scratch/ehu_p518_1/ehu_p518_1_1/Ereduak/DeepSeek-R1-Distill-Llama-70B")
parser.set_defaults(language="en")
parser.set_defaults(split="train")
parser.set_defaults(n=32)
parser.set_defaults(guidelines=True)

args = parser.parse_args()
split = args.split
language = args.language
n = args.n
predict_dir = args.predict
guidelines = args.guidelines


LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}




map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}

#path_read = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
path_write = args.out_dir
if os.path.exists(path_write):
    os.remove(path_write)

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#model_name = "/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B"
model_name = args.model_name
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,device_map="cuda")
inputs = []
pre_dicts = []



#STEP 1: Created the reasoning
inputs_all = []
templates_all = []
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
        for template in templates_list:
            if "BERT" in model_name:
                prompt = generate_prompt_encoder(pre_dict,tokenizer,template)
            else:
                prompt = generate_prompt(pre_dict,tokenizer,language,template,guidelines=guidelines)
            inputs.append(prompt)
        templates_all.append(templates_list)
        inputs_all.append(inputs)
        pre_dict_all.append(pre_dict)


max_last = 0
new_pred_dict_all = []
with torch.no_grad():
    for templates, inputs, pre_dict in zip(templates_all, inputs_all, pre_dict_all):
        best_template = []
        max_log = float("-inf")
        for inp, temp in zip(inputs, templates):
            logits = model(inp).logits
            logits_item = logits.item()
            print("Logits: ", logits_item)
            if logits_item > max_log:
                max_log = logits_item
                best_template = temp
        new_pre_dict = {
            "docid": pre_dict["docid"],
            "doctext": pre_dict["doctext"],
            "templates": pre_dict["templates"],
            "pred_json_reward": best_template,
            "logits": max_log
        }
        new_pred_dict_all.append(new_pre_dict)

# Write the results to a jsonl file
with open(path_write, 'w', encoding='utf-8') as f:
    for new_pre_dict in new_pred_dict_all:
        json_line = json.dumps(new_pre_dict, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Results written to {path_write}")




