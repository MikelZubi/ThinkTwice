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
from utils import maxCommStr
import torch

#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--split', dest='split', type=str)
parser.add_argument('--n', dest='n', type=int)
parser.add_argument('--language', dest='language', type=str)
parser.add_argument("--model-name", dest='model_name', type=str)
parser.add_argument("--out-dir", dest="out_dir", type=str)
parser.add_argument("--guided-decoding", dest="guided_decoding", action='store_true')

parser.set_defaults(model_name="/leonardo_work/EUHPC_E04_042/BaseModels/Llama-3.3-70B-Instruct")
parser.set_defaults(language="en")
parser.set_defaults(split="train")
parser.set_defaults(n=1)
parser.set_defaults(step_prompt=False)
parser.set_defaults(add_wait=0)
parser.set_defaults(guided_decoding=False)

args = parser.parse_args()
split = args.split
language = args.language
n = args.n
guided_decoding = args.guided_decoding

LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_promt(data,tokenizer,language_code):
    language = LANGUAGE_MAP[language_code]
    prompt = [{'role': 'system', 'content': PROMPT_FN["P_S_MUC_LLAMA_JSON"].format(language=language)}]
    prompt.append({"role": "user", "content": PROMPT_FN["P_U_MUC_LLAMA_JSON"].format(document=data["doctext"])})
    prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    return prompt_token_ids


map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}

path_read = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
path_write = args.out_dir
if os.path.exists(path_write):
    os.remove(path_write)

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#model_name = "/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B"
model_name = args.model_name
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.90, max_model_len=10000)
inputs = []
pre_dicts = []



#STEP 1: Created the reasoning

with open(path_read, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        inputs.append(generate_promt(pre_dict,tokenizer,language))
        pre_dicts.append(pre_dict)

if n == 1:
    temperature = 0.0
else:
    temperature = 0.7

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]
if guided_decoding:
    guided_decoding_params = GuidedDecodingParams(json=Base.model_json_schema(),backend="outlines")
    result_1 = llm.generate(
        prompt_token_ids=inputs,
        sampling_params=SamplingParams(
            temperature=temperature, #Recommended value
            max_tokens=4000,
            seed=42,
            stop_token_ids=terminators,
            n=n,
            guided_decoding=guided_decoding_params
        ),
        use_tqdm=True
    )
else:
    result_1 = llm.generate(
        prompt_token_ids=inputs,
        sampling_params=SamplingParams(
            temperature=temperature, #Recommended value
            max_tokens=4000,
            seed=42,
            stop_token_ids=terminators,
            n=n
        ),
        use_tqdm=True
    )
new_inputs = []
for idx, outputs in enumerate(result_1):
    pre_dicts[idx]["pred_reasoning"] = []
    pre_dicts[idx]["pred_json"] = []
    lower_doc = pre_dicts[idx]["doctext"].lower()
    for j, output in enumerate(outputs.outputs):
        post_templates = []
        print(output.text)
        try:
            _ = Base(**json.loads(output.text))
            for template in json.loads(output.text)["templates"]:
                post_processed = {}
                for key in template.keys():
                    if key != "incident_type" and template[key] != []:
                        post_processed[key] = []
                        for elem in template[key]:
                            lower_elem = elem.lower()
                            if lower_elem in lower_doc:
                                post_processed[key].append([lower_elem])
                            else:
                                commn_str = maxCommStr(lower_elem, lower_doc)
                                if commn_str != "":
                                    if commn_str[0] == " ":
                                        commn_str = commn_str[1:]  # Remove leading space
                                    if commn_str[-1] == " ":
                                        commn_str = commn_str[:-1]
                                    post_processed[key].append([commn_str])
                    else:
                        post_processed[key] = template[key]
                post_templates.append(post_processed)
        except:
            post_templates.append(["ERROR"])  # Only if doesn't stop generating, and reach the maximun number of tokens

        if n > 1:
            pre_dicts[idx]["pred_json"].append(post_templates)
        else:
            pre_dicts[idx]["pred_json"] = post_templates

print("Done")
with open(path_write, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



