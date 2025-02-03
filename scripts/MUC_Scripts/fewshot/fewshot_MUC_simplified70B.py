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


def generate_prompt(doc,tokenizer, language_code,k=2, random=True):
    language = LANGUAGE_MAP[language_code]
    with open("multimuc/data/multimuc_v1.0/corrected/"+language_code+"/train_simplified_preprocess.jsonl") as f:
        count = 0
        if k > 0:
            prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act). To better undestand the task you will have some few-shot information'}]
        else:
            prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act).'}]
        all_train = f.readlines()
        if random:
            rd.shuffle(all_train)
        if k > 0:
            for line in all_train:
                inputs = json.loads(line)
                template_str = json.dumps(inputs["templates"], ensure_ascii=False)
                prompt.append({"role":"user","content":inputs["doctext"]})
                prompt.append({"role":"assistant","content":template_str})
                count += 1
                if count >= k:
                    break
        prompt.append({"role":"user","content":doc})
        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt_token_ids
    


model_name = "meta-llama/Llama-3.3-70B-Instruct"
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
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

llm = LLM(model=model_name, tensor_parallel_size=2, enforce_eager=True,guided_decoding_backend="lm-format-enforcer", gpu_memory_utilization=1.0, max_model_len=70000)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
        prompt = generate_prompt(data["doctext"],tokenizer,language,k,random)
        inputs.append(prompt)
denboa1 = time.time()
print("Time taken: ", time.time()-denboa1)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
guided_decoding_params = GuidedDecodingParams(json=Base.model_json_schema(),backend="lm-format-enforcer")
print("Generating...")
result = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        guided_decoding=guided_decoding_params,
        temperature=0.0,
        max_tokens=2000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)
output = "output.txt"
with open(output, "w") as f:
    for output in result:
        f.write(output.outputs[0].text + "\n")
for idx, output in enumerate(result):
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
    folder_path = "predictions/predictions_MUC_simplified_70B/random-few/"+str(language)

else:
    folder_path = "predictions/predictions_MUC_simplified_70B/first-few/"+str(language)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
path = folder_path + "/"+str(k)+"-shot_greedy.json"
with open(path, "w") as outfile:
    json.dump(pred_dict, outfile, indent=4,ensure_ascii=False)
destroy_process_group()