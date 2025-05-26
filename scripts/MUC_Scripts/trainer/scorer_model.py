from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
import os
from torch.distributed import destroy_process_group
import argparse

import sys 
sys.path.append("class_data")
sys.path.append("prompt_library")
from MUC_Class_simplified import *
from init import PROMPT_FN
from copy import deepcopy
from collections import defaultdict



#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--read-file', dest='read_file', type=str)
parser.add_argument('--language', dest='language', type=str)
parser.add_argument("--model-name", dest='model_name', type=str)
parser.add_argument("--out-dir", dest="out_dir", type=str)
parser.add_argument("--reasoning", dest="reasoning", action='store_true')

parser.set_defaults(model_name="/leonardo_work/EUHPC_E04_042/BaseModels/Llama-3.3-70B-Instruct")
parser.set_defaults(language="en")

parser.set_defaults(split="train")
parser.set_defaults(reasoning=False)

args = parser.parse_args()
language = args.language
reasoning = args.reasoning




LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}

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

def generate_promt(data,tokenizer,language_code, reasoning):
    language = LANGUAGE_MAP[language_code]
    templates = data["pred_json"]
    template_string = ""
    template_counter = defaultdict(int)
    for template in templates:
        if template == ["ERROR"] or template == [["ERROR"]]:
            continue
        simp_template = simplify_template(template)
        template_counter[simp_template] += 1

    maximun = 0
    maximun_template = ""
    for template in template_counter:
        if template_counter[template] > maximun:
            maximun = template_counter[template]
            maximun_template = template
        template_string += str(template_counter[template]) + "- " + template + " \n"
    if maximun_template == '{"templates": []}':
        for template in template_counter:
            if template_counter[template] > maximun/2:
                maximun = template_counter[template] * 2
                maximun_template = template
    print(template_string)
    if not reasoning:
        system_prompt = PROMPT_FN["P_S_MUC_LLAMA_SCORER"].format( language=language)
        user_prompt = PROMPT_FN["P_U_MUC_LLAMA_SCORER"].format(document=data["doctext"], templates=template_string)
        prompt = [{'role': 'system', 'content': system_prompt}]
        prompt.append({'role': 'user', 'content': user_prompt})
        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    else:
        user_prompt = PROMPT_FN["P_U_MUC_70BR1_REASONING"].format(language=language, document=data["doctext"], templates=template_string)
        prompt = [{'role': 'user', 'content': user_prompt}]
        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True) + tokenizer.encode("<think>\n")

    
    return prompt_token_ids, maximun_template


map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}

path_read = args.read_file
path_write = args.out_dir
if os.path.exists(path_write):
    os.remove(path_write)

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#model_name = "/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B"
model_name = args.model_name
base_name = model_name.split("/")[-1]
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.90, max_model_len=40000)
inputs = []
pre_dicts = []



#STEP 1: Created the reasoning
max_templates = []
with open(path_read, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        current_input, max_template = generate_promt(pre_dict,tokenizer,language, reasoning)
        inputs.append(current_input)
        max_templates.append(max_template)
        pre_dicts.append(pre_dict)


if reasoning:
    terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("</think>")]
    result_1 = llm.generate(
        prompt_token_ids=inputs,
        sampling_params=SamplingParams(
            temperature=0.0,
            max_tokens=4000,
            seed=42,
            stop_token_ids=terminators,
            n=1
        ),
        use_tqdm=True
    )
    new_inputs = []
    for idx, outputs in enumerate(result_1):
        for output in outputs.outputs:
            new_inputs.append(inputs[idx] + list(output.token_ids))

else:
    new_inputs = inputs
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  


guided_decoding_params = GuidedDecodingParams(json=Base.model_json_schema(),backend="outlines")
result_2 = llm.generate(
    prompt_token_ids=new_inputs,
    sampling_params=SamplingParams(
        temperature=0.0,
        max_tokens=1000,
        stop_token_ids=terminators,
        guided_decoding=guided_decoding_params,
        n=1
    ),
    use_tqdm=True
)
for idx, outputs in enumerate(result_2):
    post_templates = []
    try:
        for template in json.loads(outputs.outputs[0].text)["templates"]:
            post_processed = {}
            for key in template.keys():
                if key != "incident_type" and template[key] != []:
                    post_processed[key]=[[elem.lower()] for elem in template[key]]
                else:
                    post_processed[key]=template[key]
            post_templates.append(post_processed)
    except:
        post_templates.append("ERROR") #Only if doesn't stop generating, and reach the maximun number of tokens
    pre_dicts[idx]["pred_json_scorer"] = post_templates
    post_templates_max = []
    for template in json.loads(max_templates[idx])["templates"]:
            post_processed_max = {}
            for key in template.keys():
                if key != "incident_type" and template[key] != []:
                    post_processed_max[key]=[[elem.lower()] for elem in template[key]]
                else:
                    post_processed_max[key]=template[key]
            post_templates_max.append(post_processed_max)
    pre_dicts[idx]["pred_json_scorer_max"] = post_templates_max


print("Done")
with open(path_write, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



