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


#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--split', dest='split', type=str)
parser.add_argument('--n', dest='n', type=int)
parser.add_argument('--language', dest='language', type=str)
parser.add_argument('--step-prompt', dest='step_prompt', action='store_true')
parser.add_argument("--model-name", dest='model_name', type=str)
parser.add_argument("--out-dir", dest="out_dir", type=str)

parser.set_defaults(model_name="/scratch/ehu_p518_1/ehu_p518_1_1/Ereduak/DeepSeek-R1-Distill-Llama-70B")
parser.set_defaults(language="en")
parser.set_defaults(split="train")
parser.set_defaults(n=32)
parser.set_defaults(step_prompt=False)

args = parser.parse_args()
split = args.split
language = args.language
n = args.n



LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_promt(data,tokenizer,language_code,step_prompt=False):
    language = LANGUAGE_MAP[language_code]
    if step_prompt:
        user_prompt = PROMPT_FN["P_U_MUC_70BR1_STEPS_REASONING"].format(language=language, document=data["doctext"])
    else:
        user_prompt = PROMPT_FN["P_U_MUC_70BR1_REASONING"].format(language=language, document=data["doctext"])
    prompt = [{'role': 'user', 'content': user_prompt}]
    prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True) + tokenizer.encode("<think>\n")
    return prompt_token_ids


map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}

path_read = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
step_prompt = args.step_prompt
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
        inputs.append(generate_promt(pre_dict,tokenizer,language,step_prompt))
        pre_dicts.append(pre_dict)

if n == 1:
    temperature = 0.0
else:
    temperature = 0.7
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]
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

    for output in outputs.outputs:
        splited_text = output.text.split("</think>")
        pre_dicts[idx]["pred_reasoning"].append(splited_text[0])
        post_templates=[]
        try:
            _ = Base(**json.loads(splited_text[1]))
            for template in json.loads(splited_text[1])["templates"]:
                post_processed = {}
                for key in template.keys():
                    if key != "incident_type" and template[key] != []:
                        post_processed[key] = [[elem] for elem in template[key]]
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



