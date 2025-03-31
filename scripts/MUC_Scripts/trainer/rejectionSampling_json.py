from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
from transformers import set_seed
import os
from torch.distributed import destroy_process_group
import argparse

import sys 
sys.path.append("class_data")
from MUC_Class_simplified import *


#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--split', dest='split', type=str)
parser.add_argument('--n', dest='n', type=int)
parser.add_argument('--language', dest='language', type=str)
parser.set_defaults(language="en")
parser.set_defaults(split="train")
parser.set_defaults(n=32)
args = parser.parse_args()



LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_prompt(data,tokenizer, language_code):
    doc = data["doctext"]
    language = LANGUAGE_MAP[language_code]
    prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act).'}]
    prompt.append({"role":"user","content":doc})
    prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    return prompt_token_ids

map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}
split = args.split
language = args.language
n = args.n
path_read = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
path_write = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_rejectionSampling_JSON"+str(n)+".jsonl"
if os.path.exists(path_write):
    os.remove(path_write)

#model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
model_name = "meta-llama/Llama-3.3-70B-Instruct"
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.90, max_model_len=40000)
inputs = []
pre_dicts = []



#STEP 1: Created the reasoning

with open(path_read, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        inputs.append(generate_prompt(pre_dict,tokenizer,language))
        pre_dicts.append(pre_dict)


terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
guided_decoding_params = GuidedDecodingParams(json=Base.model_json_schema(),backend="outlines")
result_1 = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        temperature=0.7, #Recommended value
        max_tokens=1000,
        stop_token_ids=terminators,
        guided_decoding=guided_decoding_params,
        n=n
    ),
    use_tqdm=True
)
new_inputs = []
for idx, outputs in enumerate(result_1):
    pre_dicts[idx]["pred_json"] = []
    for output in outputs.outputs:
        post_templates = []
        try:
            for template in json.loads(output.text)["templates"]:
                post_processed = {}
                for key in template.keys():
                    if key != "incident_type" and template[key] != []:
                        post_processed[key]=[[elem] for elem in template[key]]
                    else:
                        post_processed[key]=template[key]
                post_templates.append(post_processed)
        except:
            post_templates.append("ERROR") #Only if doesn't stop generating, and reach the maximun number of tokens
        pre_dicts[idx]["pred_json"].append(post_templates)



with open(path_write, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



