import sys 
sys.path.append("class_data")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
import time
from transformers import set_seed
import random as rd
import os
from torch.distributed import destroy_process_group
import sys
from MUC_Class_simplified import *
import argparse


LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_prompt(doc,tokenizer, language_code, reasoning=False):
    language = LANGUAGE_MAP[language_code]
    if reasoning:
        prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+'. For that, you need to follow two steps. The first one is the following: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage". The second and last step is to extract the entities of the document that take part on the incident types that you have extracted in the previous step. The entities can be of the following types: "A person responsible for the incident (PerpInd)", "An organization responsible for the incident (PerpOrg)", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack (Victim)", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)". After that, you need to create a JSON that stores the information that you have extracted.'}]
    else:
        prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act).'}]
    prompt.append({"role":"user","content":doc})
    prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    return prompt_token_ids
    
def generate_text_train(doc, tokenizer,language_code, reasoning=False):
    language = LANGUAGE_MAP[language_code]
    if reasoning:
        prompt = '<USER> You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+'. For that, you need to follow two steps. The first one is the following: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage". The second and last step is to extract the entities of the document that take part on the incident types that you have extracted in the previous step. The entities can be of the following types: "A person responsible for the incident (PerpInd)", "An organization responsible for the incident (PerpOrg)", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack (Victim)", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)". After that, you need to create a JSON that stores the information that you have extracted. The document is the next one:\n"' + doc + '" </USER>\n'
    else:
        prompt = '<USER> You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act). The document is the next one:\n"' + doc + '" </USER>\n'
    prompt += "<ASSISTANT> "
    prompt_token_ids = tokenizer.encode(prompt)
    return prompt_token_ids
    

#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the SFT trainer')
parser.add_argument('--reasoning', dest='reasoning', action='store_true',
                    help='Use reasoning.')
parser.add_argument('--natural-reasoning', dest='natural_reasoning', action='store_true',
                    help='Use natural reasoning, default to artificial.')
parser.add_argument('--random-decoding', dest='random_decoding', action='store_true',
                    help='Use beam search.')
parser.add_argument("--GRPO", dest="GRPO", action="store_true", help="Use GRPO.")
parser.add_argument("--r1", dest="r1", action="store_true", help="Use Llama 3.1 8B destill DeepSeek R1")
parser.set_defaults(reasoning=False)
parser.set_defaults(natural_reasoning=False)
parser.set_defaults(random_decoding=False)
parser.set_defaults(GRPO=False)
args = parser.parse_args()

#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"

if args.reasoning and not args.natural_reasoning:
    model_path = "Model_Reasoning"
elif args.reasoning and args.natural_reasoning:
    model_path = "Model_Natural_Reasoning"
else:
    model_path = "Model_JSON"

if args.GRPO:
    model_path = "Model_GRPO_" + model_path[6:]

if args.r1:
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model_path = model_path + "_R1"
else:
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
language = "en"
seed = 42
set_seed(seed)
rd.seed(seed)

inputs = []
docids = []
pred_dict = {}

with open("multimuc/data/multimuc_v1.0/corrected/"+language+"/test.jsonl") as f:
    for line in f:
        data = json.loads(line)
        docid = str(
                int(data["docid"].split("-")[0][-1]) * 10000
                + int(data["docid"].split("-")[-1])
            )
        docids.append(docid)
        pred_dict[docid] = {}
        pred_dict[docid]["doctext"] = data["doctext"]
        pred_dict[docid]["gold_templates"] = data["templates"]
        if args.r1:
            prompt = generate_text_train(data["doctext"],tokenizer,language,reasoning=args.reasoning)
        else:
            prompt = generate_prompt(data["doctext"],tokenizer,language,reasoning=args.reasoning)
        inputs.append(prompt)
denboa1 = time.time()
llm = LLM(model=model_path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.8)
print("Time taken: ", time.time()-denboa1)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]    
guided_decoding_params = GuidedDecodingParams(json=Base.model_json_schema(),backend="lm-format-enforcer")
print("Generating...")
if args.random_decoding:
    temperature = 0.5
else:
    temperature = 0.0
result = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        #guided_decoding=guided_decoding_params,
        temperature=temperature,
        max_tokens=4000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)

for idx, output in enumerate(result):
    try:
        post_templates = []
        if args.reasoning:
            splited_text = output.outputs[0].text.split('{"templates":')
            out_text = '{"templates":' + splited_text[-1]
            print(output.outputs[0].text)
            pred_dict[docids[idx]]["reasoning"] = output.outputs[0].text
        else:
            out_text = output.outputs[0].text
            print(out_text)
        if args.r1:
            out_text = out_text.split("</ASSISTANT>")[0]
        for template in json.loads(out_text)["templates"]:
            post_processed = {}
            for key in template.keys():
                if key != "incident_type" and template[key] != []:
                    post_processed[key]=[[elem] for elem in template[key]]
                else:
                    post_processed[key]=template[key]
            post_templates.append(post_processed)
        pred_dict[docids[idx]]["pred_templates"] = post_templates

    except json.decoder.JSONDecodeError:
        print("ERROR")
        pred_dict[docids[idx]]["pred_templates"] = []



if args.reasoning and not args.natural_reasoning:
    folder_path = "predictions/MUC_simplified_SFT_Reasoning"
elif args.reasoning and args.natural_reasoning:
    folder_path = "predictions/MUC_simplified_SFT_Natural_Reasoning"
else:
    folder_path = "predictions/MUC_simplified_SFT_JSON"

if args.GRPO:
    folder_path = folder_path.replace("SFT","GRPO")
if args.r1:
    folder_path = folder_path + "R1"

folder_path = folder_path + "/" + str(language)


if not os.path.exists(folder_path):
    os.makedirs(folder_path)
if not args.random_decoding:
    path = folder_path + "/greedy.json"
else:
    path = folder_path + "/random_decoding.json"
with open(path, "w") as outfile:
    json.dump(pred_dict, outfile, indent=4,ensure_ascii=False)
destroy_process_group()