from transformers import AutoTokenizer
from vllm import LLM, SamplingParams, inputs
from vllm.sampling_params import GuidedDecodingParams
import json
import os
from torch.distributed import destroy_process_group
import argparse


import sys 
sys.path.append("class_data")
from BETTER_Granular_Class_string import *
sys.path.append("inference_library")
from prompt_factory import prompt_factory
import hyperparameters
from copy import deepcopy
from utils import maxCommStr



#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--split', dest='split', type=str)
parser.add_argument('--n', dest='n', type=int)
parser.add_argument('--language', dest='language', type=str)
parser.add_argument("--model-name", dest='model_name', type=str)
parser.add_argument("--out-dir", dest="out_dir", type=str)
parser.add_argument("--add-wait", dest="add_wait", type=int)
parser.add_argument("--think", dest="think", action='store_true')
parser.add_argument("--shots", dest="shots", type=int, default=0) #TODO: not implemented yet

parser.set_defaults(model_name="/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B")
parser.set_defaults(language="en")
parser.set_defaults(split="train")
parser.set_defaults(n=32)
parser.set_defaults(think=False)
parser.set_defaults(add_wait=0)

args = parser.parse_args()
split = args.split
language = args.language
n = args.n
model_name = args.model_name
think = args.think
path_write = args.out_dir
add_wait = args.add_wait




#Inferencen Hyperparameters
hyperparameters = hyperparameters.Hyperparameters(model_name, think, n)
temperature = hyperparameters.temperature
top_p = hyperparameters.top_p
top_k = hyperparameters.top_k
min_p = hyperparameters.min_p
print(f"Temperature: {temperature}, Top_p: {top_p}, Top_k: {top_k}, Min_p: {min_p}")


LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}
language_name = LANGUAGE_MAP[language] 

prompt = prompt_factory(model_name, language_name, "BETTER", think)






map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}

path_read = f"phase2/phase2.granular.eng.preprocess-{split}-simplified.jsonl"
if os.path.exists(path_write):
    os.remove(path_write)


tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.90, max_model_len=32768)
input_ids = []
pre_dicts = []



with open(path_read, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = {"docid": data["docid"], "doctext": data["doctext"]}
        input_ids.append(inputs.TokensPrompt({"prompt_token_ids":prompt.generate_prompt(data)}))
        pre_dicts.append(pre_dict)


terminators = [
    tokenizer.eos_token_id,
    #tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("</think>")]  
if think:
    result_1 = llm.generate(
        input_ids,
        sampling_params=SamplingParams(
            #Recommended values
            temperature=temperature, 
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            max_tokens=4000,
            seed=42,
            stop_token_ids=terminators,
            n=n,

        ),
        use_tqdm=True
    )
    #TODO: not implemented for n > 1
    if add_wait >= 1:
        reasoning = []
        for i in range(add_wait):
            new_inputs = []
            lag_input = deepcopy(input_ids)
            for w, ids in enumerate(lag_input):
                if i == 0:
                    reasoning.append([])
                for j in range(n):
                    new_input = ids + list(result_1[w].outputs[j].token_ids[:-2]) + list(tokenizer.encode(" Wait")[1:])
                    #new_input = input + list(result_1[w].outputs[j].token_ids[:-1]) + [tokenizer.encode("Wait")[1]]
                    new_inputs.append(inputs.TokensPrompt({"prompt_token_ids": new_input}))
                    
                    if i != 0:
                        reasoning[w][j] += result_1[w].outputs[j].text + " Wait"
                    else:
                        reasoning[w].append(result_1[w].outputs[j].text + " Wait")
            input_ids = new_inputs
            

            result_1 = llm.generate(
                prompt_token_ids=input_ids,
                sampling_params=SamplingParams(
                    temperature=temperature,
                    max_tokens=2000,
                    stop_token_ids=terminators,
                    n=1,
                    seed=42
                ),
                use_tqdm=True
            )
    new_inputs = []
    for idx, outputs in enumerate(result_1):
        if add_wait > 0:
            pre_dicts[idx]["pred_reasoning"] = [reasoning[idx][j] + output.text for j, output in enumerate(outputs.outputs)]
        else:
            pre_dicts[idx]["pred_reasoning"] = [output.text for output in outputs.outputs]
        pre_dicts[idx]["templates"] = [None]*n
        for output in outputs.outputs:
            lag_ids = input_ids[idx]["prompt_token_ids"] + list(output.token_ids)
            new_inputs.append(inputs.TokensPrompt({"prompt_token_ids": lag_ids}))
else:
    new_inputs = []
    for idx, ids in enumerate(input_ids):
        for j in range(n):
            new_inputs.append(ids)
        pre_dicts[idx]["templates"] = [None]*n


terminators = [
    tokenizer.eos_token_id,
    #tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

structured_outputs = GuidedDecodingParams(json=Template.model_json_schema(),backend="outlines")
result_2 = llm.generate(
    new_inputs,
    sampling_params=SamplingParams(
        temperature=temperature, 
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=2000,
        stop_token_ids=terminators,
        structured_outputs=structured_outputs,
        seed=42,
        n=1
    ),
    use_tqdm=True
)
for idx_n, outputs in enumerate(result_2):
    idx = idx_n // n
    post_templates = []
    lower_doc = pre_dicts[idx]["doctext"].lower()
    try:
        post_templates = json.loads(outputs.outputs[0].text)["templates"]
    except json.decoder.JSONDecodeError:
        print(outputs.outputs[0].text)
        post_templates = ["ERROR"]
    if n > 1:
        pre_dicts[idx]["templates"][idx_n % n] = post_templates
    else:
        pre_dicts[idx]["templates"] = post_templates


print("Done")
with open(path_write, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



