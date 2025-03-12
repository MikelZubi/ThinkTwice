from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from vllm import LLM, SamplingParams
import torch
import json
import argparse
import os
import shutil
import random as rd
from torch.distributed import destroy_process_group
import copy as cp

LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


ALL_KEYS = set(["incident_type", "PerpInd", "PerpOrg", "Target", "Victim", "Weapon"])
def ensure_format(pred):
    try:
        pred_json = json.loads(pred)
        if type(pred_json) != dict:
            return False
        if pred_json["templates"] != []:
            for template in pred_json["templates"]:
                if type(template) != dict:
                    return False
                elif not ALL_KEYS.issubset(template.keys()):
                    return False
    except (json.JSONDecodeError, KeyError):
        return False
    return True

#With llama3 Instruct model
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
        prompt = '<USER> You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+'. For that, you need to follow two steps. The first one is the following: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage". The second and last step is to extract the entities of the document that take part on the incident types that you have extracted in the previous step. The entities can be of the following types: "A person responsible for the incident (PerpInd)", "An organization responsible for the incident (PerpOrg)", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack (Victim)", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)". After that, you need to create a JSON that stores the information that you have extracted. The document is the next one:\n"' + doc +'" </USER>\n'
    else:
        prompt = '<USER> You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act). The document is the next one:\n"' + doc + '" </USER>\n'
    prompt += "<ASSISTANT> "
    prompt_token_ids = tokenizer.encode(prompt)
    return prompt_token_ids
    
def convert_docid(docid: str) -> str:
    return str(int(docid.split("-")[0][-1]) * 10000 + int(docid.split("-")[-1]))

 #Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the SFT trainer')
parser.add_argument('--reasoning', dest='reasoning', action='store_true',
                    help='Use reasoning.')
parser.add_argument('--natural-reasoning', dest='natural_reasoning', action='store_true',
                    help='Use natural reasoning, default to artificial.')
parser.add_argument('--model-path', dest="model_path", type=str)
parser.add_argument('--model-name', dest="model_name", type=str)
parser.set_defaults(reasoning=False)
parser.set_defaults(natural_reasoning=False)
parser.set_defaults(model_path='meta-llama/Meta-Llama-3.1-8B-Instruct')
parser.set_defaults(model_name='meta-llama/Meta-Llama-3.1-8B-Instruct')
args = parser.parse_args()

#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model_name = args.model_name

pred_dict = {}
docids = []
inputs = []
language = "en"
seed = 42
set_seed(seed)
chkpt_path= args.model_path
chkpt = chkpt_path.split("/")[-1]
merge_path = "Model_Merged"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype= torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Path to save the merged model
# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, chkpt_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merge_path,safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(merge_path)
with open("multimuc/data/multimuc_v1.0/corrected/"+language+"/dev.jsonl") as f:
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
        if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
            prompt = generate_prompt(data["doctext"],tokenizer,language,reasoning=args.reasoning)
        else:
            prompt = generate_text_train(data["doctext"],tokenizer,language,reasoning=args.reasoning)
        inputs.append(prompt)

llm = LLM(model=merge_path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.8)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
print("Generating...")
result = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        #guided_decoding=guided_decoding_params,
        temperature=0.0,
        max_tokens=4000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)
for idx, output in enumerate(result):
    try:
        post_templates = []
        if args.reasoning:
            out_text = '{"templates":' + output.outputs[0].text.split('{"templates":')[-1]
        else:
            out_text = output.outputs[0].text
        if model_name != "meta-llama/Meta-Llama-3.1-8B-Instruct":
            out_text = out_text.split("</ASSISTANT>")[0]
        loaded_json = json.loads(out_text)
        if not ensure_format(out_text):
            print(out_text)
            print("ERROR")
            pred_dict[docids[idx]]["pred_templates"] = []
            continue
        for template in loaded_json["templates"]:
            post_processed = {}
            for key in template.keys():
                if key != "incident_type" and template[key] != []:
                    post_processed[key]=[[elem] for elem in template[key]]
                else:
                    post_processed[key]=template[key]
            post_templates.append(post_processed)
        pred_dict[docids[idx]]["pred_templates"] = post_templates

    except (json.decoder.JSONDecodeError, KeyError) as e:
        print(out_text)
        print("ERROR")
        pred_dict[docids[idx]]["pred_templates"] = []



if args.reasoning and not args.natural_reasoning:
    folder_path = "predictions/DEV/MUC_simplified_SFT_Reasoning"
elif args.reasoning and args.natural_reasoning:
    folder_path = "predictions/DEV/MUC_simplified_SFT_Natural_Reasoning"
else:
    folder_path = "predictions/DEV/MUC_simplified_SFT_JSON"

if model_name != "meta-llama/Meta-Llama-3.1-8B-Instruct":
    folder_path = folder_path + "R1"

folder_path = folder_path + "/" + str(language)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
path = folder_path + "/"+chkpt+".json"
with open(path, "w") as outfile:
    json.dump(pred_dict, outfile, indent=4,ensure_ascii=False)
destroy_process_group()
shutil.rmtree(merge_path, ignore_errors=True)
        
    
