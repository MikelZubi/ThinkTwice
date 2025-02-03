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


LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_prompt(data,tokenizer, language_code,step, k=2, random=True):
    language = LANGUAGE_MAP[language_code]
    with open("multimuc/data/multimuc_v1.0/corrected/"+language_code+"/train_simplified_preprocess.jsonl") as f:
        count = 0
        if step == 1:
            prompt =[{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' in JSON format. For that, you need to follow a number of steps. This is the first one: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage".'}]
        elif step == 2:
            prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' in JSON format. For that, you need to follow a number of steps. This is the second one: you need to extract the entities of the document that take part on the incident types that you have already extracted. The entities can be of the following types: "A person responsible for the incident", "An organization responsible for the incident", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)".'}]
        elif step == 3:
            with open(os.path.join("Docs", "MUC_simplified.md"), 'r') as md_file:
                guidelines =  md_file.read()
            prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' in JSON format. For that, you need to follow a number of steps. This is the last one: You need to fill some templates with the information of the next document. The guidelines of the template are the following:\n'+ guidelines +'\n To do it, you have already extracted in previous steps what are the entities and the incident types of the document, you need to use that information to fill the templates.'}]
        if k > 0:
            prompt[0]["content"]+= " To better undestand the task you will have some few-shot information."
        all_train = f.readlines()
        if random:
            rd.shuffle(all_train)
        if k > 0:
            for line in all_train:
                inputs = json.loads(line)
                if step == 1:
                    incident_types_str = json.dumps(inputs["incident_types"], ensure_ascii=False)
                    prompt.append({"role":"user","content":"The document is the next one: \n '" + inputs["doctext"] + "' \n Identify what incident types are happening in it."})
                    prompt.append({"role":"assistant","content":incident_types_str})
                if step == 2:
                    entities_str = json.dumps(inputs["entities"], ensure_ascii=False)
                    len_incident_types = str(len(inputs["incident_types"]["incident_types"]))
                    incident_types_str = json.dumps(inputs["incident_types"], ensure_ascii=False)
                    prompt.append({"role":"user","content":"The document is the next one: \n '" + inputs["doctext"] + "' \n From it you have already extracted the following " + len_incident_types + " incident types: \n '" + incident_types_str + "' \n Now you need to extract the entities that take part on those incident types."})
                    prompt.append({"role":"assistant","content":entities_str})
                if step == 3:
                    entities_str = json.dumps(inputs["entities"], ensure_ascii=False)
                    len_incident_types = str(len(inputs["incident_types"]["incident_types"]))
                    incident_types_str = json.dumps(inputs["incident_types"], ensure_ascii=False)
                    template_str = json.dumps(inputs["templates"], ensure_ascii=False)
                    prompt.append({"role":"user","content":"The document is the next one: \n '" + inputs["doctext"] + "' \n From it you have already extracted the following " + len_incident_types + " incident types: \n '" + incident_types_str + "' \n You have also extracted the entities that take part on those incident types: \n '" + entities_str + "'\n Now you need to fill that entities into a template for each incident type."})
                    prompt.append({"role":"assistant","content":template_str})
                count += 1
                if count >= k:
                    break
        if step == 1:
            prompt.append({"role":"user","content":"The document is the next one: \n '" + data["doctext"] + "' \n Identify what incident types are happening in it."})
        if step == 2:
            len_incident_types = str(len(data["pred_incident_types"]["incident_types"]))
            incident_types_str = json.dumps(data["pred_incident_types"], ensure_ascii=False)
            prompt.append({"role":"user","content":"The document is the next one: \n '" + data["doctext"] + "' \n From it you have already extracted the following " + len_incident_types + " incident types: \n '" + incident_types_str + "' \n Now you need to extract the entities that take part on those incident types."})
        if step == 3:
            len_incident_types = str(len(data["pred_incident_types"]["incident_types"]))
            incident_types_str = json.dumps(data["pred_incident_types"], ensure_ascii=False)
            entities_str = json.dumps(data["pred_entities"], ensure_ascii=False)
            prompt.append({"role":"user","content":"The document is the next one: \n '" + data["doctext"] + "' \n From it you have already extracted the following " + len_incident_types + " incident types: \n '" + incident_types_str + "' \n You have also extracted the entities that take part on those incident types: \n '" + entities_str + "'\n Now you need to fill that entities into a template for each incident type."})
        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt_token_ids
    


model_name = "meta-llama/Llama-3.2-3B-Instruct"
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
k = int(sys.argv[1])
if sys.argv[2] == "R":
    random = True
else:
    random = False
language = str(sys.argv[3])
seed = 42
set_seed(seed)
rd.seed(seed)

inputs = []
docids = []
pred_dict = {}

#STEP 1
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
        #pred_dict[docid]["gold_entities"] = data["entities"]
        #pred_dict[docid]["gold_incident_types"] = data["incident_types"]
        prompt = generate_prompt(pred_dict[docid],tokenizer,language,1,k,random)
        inputs.append(prompt)
denboa1 = time.time()
llm = LLM(model=model_name, tensor_parallel_size=1, enforce_eager=True,guided_decoding_backend="lm-format-enforcer")
print("Time taken: ", time.time()-denboa1)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
guided_decoding_params_1 = GuidedDecodingParams(json=Incident_Types.model_json_schema(),backend="lm-format-enforcer")
result_1 = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        guided_decoding=guided_decoding_params_1,
        temperature=0.0,
        max_tokens=500,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)

for idx, output in enumerate(result_1):
    try:
        incident_types = json.loads(output.outputs[0].text)
        pred_dict[docids[idx]]["pred_incident_types"] = incident_types
    except json.decoder.JSONDecodeError:
        pred_dict[docids[idx]]["pred_incident_types"] = {"incident_types": []}


#STEP 2
inputs.clear()
with open("multimuc/data/multimuc_v1.0/corrected/"+language+"/dev.jsonl") as f:
    for line in f:
        data = json.loads(line)
        docid = str(
                int(data["docid"].split("-")[0][-1]) * 10000
                + int(data["docid"].split("-")[-1])
            )
        prompt = generate_prompt(pred_dict[docid],tokenizer,language,2,k,random)
        inputs.append(prompt)
guided_decoding_params_2 = GuidedDecodingParams(json=Entities.model_json_schema(),backend="lm-format-enforcer")

result_2 = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        guided_decoding=guided_decoding_params_2,
        temperature=0.0,
        max_tokens=500,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)
for idx, output in enumerate(result_2):
    try:
        entities = json.loads(output.outputs[0].text)
        pred_dict[docids[idx]]["pred_entities"] = entities
    except json.decoder.JSONDecodeError:
        pred_dict[docids[idx]]["pred_entities"] = {"entities": []}

    
#STEP 3
inputs.clear()
with open("multimuc/data/multimuc_v1.0/corrected/"+language+"/dev.jsonl") as f:
    for line in f:
        data = json.loads(line)
        docid = str(
                int(data["docid"].split("-")[0][-1]) * 10000
                + int(data["docid"].split("-")[-1])
            )
        prompt = generate_prompt(pred_dict[docid],tokenizer,language,3,k,random)
        inputs.append(prompt)
guided_decoding_params_3 = GuidedDecodingParams(json=Base.model_json_schema(),backend="lm-format-enforcer")
result_3 = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        guided_decoding=guided_decoding_params_3,
        temperature=0.0,
        max_tokens=2000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)


for idx, output in enumerate(result_3):
    try:
        post_templates = []
        for template in json.loads(output.outputs[0].text)["templates"]:
            post_processed = {}
            for key in template.keys():
                if key != "incident_type" and template[key] != []:
                    post_processed[key]=[[elem] for elem in template[key]]
                else:
                    post_processed[key]=template[key]
            post_templates.append(post_processed)
        pred_dict[docids[idx]]["pred_templates"] = post_templates

    except json.decoder.JSONDecodeError:
        pred_dict[docids[idx]]["pred_templates"] = []

if random:
    folder_path = "predictions/predictions_MUC_simplified_steps_3B/random-few/"+str(language)

else:
    folder_path = "predictions/predictions_MUC_simplified_steps_3B/first-few/"+str(language)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
path = folder_path + "/"+str(k)+"-shot_greedy.json"
with open(path, "w") as outfile:
    json.dump(pred_dict, outfile, indent=4,ensure_ascii=False)
destroy_process_group()