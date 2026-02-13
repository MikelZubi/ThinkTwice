import json
import os
import argparse

import sys
sys.path.append("class_data")
sys.path.append("inference_library")
from prompt_factory import prompt_factory
from vllm import LLM, inputs

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
                    new_template[key].append(element[0])
        new_templates.append(new_template)
    return json.dumps(new_templates, ensure_ascii=False)

def remove_errors(all_templates):
    return [template  if ["ERROR"]  != template and [["ERROR"]] != template and "ERROR" not in template else [] for template in all_templates]
#def remove_errors(all_templates):
#    return [template for template in all_templates if ["ERROR"]  != template and [["ERROR"]] != template and "ERROR" not in template]

#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--predict-dir', dest='predict_dir', type=str)
parser.add_argument('--language', dest='language', type=str)
parser.add_argument("--model-dir", dest='model_dir', type=str)
parser.add_argument("--out-dir", dest="out_dir", type=str)
parser.add_argument("--split", dest="split", type=str)

parser.set_defaults(model_dir="/scratch/ehu_p518_1/ehu_p518_1_1/Ereduak/DeepSeek-R1-Distill-Llama-70B")
parser.set_defaults(language="en")
parser.set_defaults(split="test")

args = parser.parse_args()
language = args.language
predict_dir = args.predict_dir
model_dir = args.model_dir
path_write = args.out_dir




LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}
language_name = LANGUAGE_MAP[language]

prompt_class = prompt_factory(model_dir, language_name, "MUC", think=False)



map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}

if os.path.exists(path_write):
    os.remove(path_write)

llm = LLM(model=model_dir, runner="pooling", enforce_eager=True, pooler_config={"activation": False, "softmax": False})
pre_dicts = []



prompt_all = []
gold_inputs =  []
templates_all = []
score_dict_all = []
pre_dict_all = []
with open(predict_dir, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        templates = remove_errors(pre_dict["pred_json"])
        templates_list = []
        for template in templates:
            if template not in templates_list:
                templates_list.append(template)
        prompts = []
        score_dict = {}
        for template in templates_list:
            str_template = json.dumps(template, ensure_ascii=False)
            score_dict[str_template] = 0.0
            simplified_templated = simplify_template(template)
            prompt = inputs.TokensPrompt({"prompt_token_ids":prompt_class.generate_prompt(pre_dict, template=simplified_templated)})
            prompts.append(prompt)
        gold_simp = simplify_template(pre_dict["templates"])
        prompt_gold = inputs.TokensPrompt({"prompt_token_ids":prompt_class.generate_prompt(pre_dict, template=gold_simp)})
        gold_inputs.append(prompt_gold)
        templates_all.append(templates_list)
        prompt_all.append(prompts)
        pre_dict_all.append(pre_dict)
        score_dict_all.append(score_dict)

prompts_flatten = []
for prompts in prompt_all:
    prompts_flatten += prompts

print("Processing predictions")
outputs_flatten = []
results = llm.classify(prompts_flatten)
for out in results:
    score = out.outputs.probs[0]
    print(score)
    outputs_flatten.append(score)
outputs_all = []
prev_index = 0
for prompt in prompt_all:
    length_inp = len(prompt)
    outputs_batch = outputs_flatten[prev_index:prev_index+length_inp]
    outputs_all.append(outputs_batch)
    prev_index += length_inp

print("Processing golds")
gold_outputs_all = []
results = llm.classify(gold_inputs)
for out in results:
    score = out.outputs.probs[0]
    print(score)
    gold_outputs_all.append(score)

new_pred_dict_all = []
for templates, outputs, pre_dict, score_dict, gold_output in zip(templates_all, outputs_all, pre_dict_all, score_dict_all, gold_outputs_all):
    best_template = []
    max_log = float("-inf")
    for output, temp in zip(outputs, templates):
        str_template = json.dumps(temp, ensure_ascii=False)
        score_dict[str_template] = output
        if output > max_log:
            max_log = output
            best_template = temp
    new_pre_dict = {
        "docid": pre_dict["docid"],
        "doctext": pre_dict["doctext"],
        "templates": pre_dict["templates"],
        "pred_json": best_template,
        "score_dict": score_dict,
        "score_gold": gold_output
    }
    new_pred_dict_all.append(new_pre_dict)

# Write the results to a jsonl file
with open(path_write, 'w', encoding='utf-8') as f:
    for new_pre_dict in new_pred_dict_all:
        json_line = json.dumps(new_pre_dict, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"Results written to {path_write}")




