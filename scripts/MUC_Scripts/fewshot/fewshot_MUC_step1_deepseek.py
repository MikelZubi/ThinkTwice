import sys 
sys.path.append("../class")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
import time
from transformers import set_seed
import random as rd
import os
import sys
from MUC_Class_simplified import *




LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_prompt(data,tokenizer, language_code,step, k=2, random=True):
    language = LANGUAGE_MAP[language_code]
    with open("multimuc/data/multimuc_v1.0/corrected/"+language_code+"/train_simplified_CoT.jsonl") as f:
        count = 0
        if step == 1: #DeepSeek
            prompt = [{'role': 'user', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+'. For that, you need to follow two steps. The first one is the following: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage". The second and last step is to extract the entities of the document that take part on the incident types that you have extracted in the previous step. The entities can be of the following types: "A person responsible for the incident (PerpInd)", "An organization responsible for the incident (PerpOrg)", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack (Victim)", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)".'}]
        elif step == 2: #Original Llama 70B
            with open(os.path.join("Docs", "MUC_simplified.md"), 'r') as md_file:
                guidelines =  md_file.read()
            prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you have already extracted the information of an document in '+language+' using some reasoning steps. Now using the information of the reasoning steps that are provided you need to fill some templates in JSON format. For that the guildelines are the followings:\n'+ guidelines +'\n'}]
        if k > 0:
            prompt[0]["content"]+= " To better undestand the task you will have some few-shot information."
        all_train = f.readlines()
        if random:
            rd.shuffle(all_train)
        if k > 0:
            for line in all_train:
                inputs = json.loads(line)
                if step == 1:
                    if count == 0:
                        prompt[0]["content"]+= "The document is the following: \n" + inputs["doctext"]
                    else:
                        prompt.append({"role":"user","content":"The document is the following: \n" + inputs["doctext"]})
                    prompt.append({"role":"assistant","content":inputs["corrected_reasoning"]})
                elif step == 2:
                    template_str = json.dumps(inputs["templates"], ensure_ascii=False)
                    prompt.append({"role":"user","content":inputs["corrected_reasoning"]})
                    prompt.append({"role":"assistant","content":template_str})
                if step == 4: 
                    continue
                count += 1
                if count >= k:
                    break
        if step == 1:
            prompt.append({"role":"user","content":"The document is the following: \n" + data["doctext"]})
        elif step == 2:
            prompt.append({"role":"user","content":data["pred_reasoning"]})
        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt_token_ids
    


#model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
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

#STEP 1: Reasoning
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
        prompt = generate_prompt(pred_dict[docid],tokenizer,language,1,k,random)
        inputs.append(prompt)
denboa1 = time.time()
llm = LLM(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=1.0, max_model_len=60000, enforce_eager=True)
print("Time taken: ", time.time()-denboa1)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
result_1 = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        temperature=0.0,
        max_tokens=2000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)

for idx, output in enumerate(result_1):
    pred_dict[docids[idx]]["pred_reasoning"] = output.outputs[0].text



if random:
    folder_path = "predictions/predictions_MUC_simplified_steps_deepseek/random-few/"+str(language)

else:
    folder_path = "predictions/predictions_MUC_simplified_steps_deepseek/first-few/"+str(language)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
path = folder_path + "/"+str(k)+"-shot_greedy.json"
with open(path, "w") as outfile:
    json.dump(pred_dict, outfile, indent=4,ensure_ascii=False)