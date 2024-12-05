
import vllm
from transformers import AutoTokenizer
from outlines.integrations.vllm import JSONLogitsProcessor
import json
import time
from transformers import set_seed
import sys
import random as rd
import os
from torch.distributed import destroy_process_group
from BETTER_Granular_Class import Template




def generate_prompt(doc,tokenizer,k=2, random=True):
    with open("phase2/phase2.granular.eng.preprocess-train.jsonl") as f:
        count = 0
        if k > 0:
            prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided as a template in JSON format. To better undestand the task you will have some few-shot information'}]
        else:
            prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided as a template in JSON format.'}]
        all_train = f.readlines()
        if random:
            rd.shuffle(all_train)
        if k > 0:
            for line in all_train:
                template_str = line.split('{"templates": ')[1][:-2]
                inputs = json.loads(line)
                prompt.append({"role":"user","content":inputs["doctext"]})
                prompt.append({"role":"assistant","content":template_str})
                count += 1
                if count >= k:
                    break
        prompt.append({"role":"user","content":doc})
        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt_token_ids
    


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
k = int(sys.argv[1])
if sys.argv[2] == "R":
    random = True
else:
    random = False
#language = str(sys.argv[3])
seed = 42
set_seed(seed)
rd.seed(seed)

inputs = []
docids = []
pred_dict = {}

with open("phase2/phase2.granular.eng.preprocess-dev.jsonl") as f:
    for line in f:
        data = json.loads(line)
        docid = data["docid"]
        docids.append(docid)
        pred_dict[docid] = {}
        pred_dict[docid]["doctext"] = data["doctext"]
        pred_dict[docid]["gold_templates"] = data["templates"]
        prompt = generate_prompt(data["doctext"],tokenizer,k,random)
        inputs.append(prompt)
denboa1 = time.time()
llm = vllm.LLM(model=model_name, tensor_parallel_size=1, enforce_eager=True)
print("Time taken: ", time.time()-denboa1)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
logits_processor = JSONLogitsProcessor(schema=Template, llm=llm, whitespace_pattern=r" ?")
print("Generating...")
result = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=vllm.SamplingParams(
        logits_processors=[logits_processor],
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
        pred_dict[docids[idx]]["pred_templates"] = json.loads(output.outputs[0].text)["templates"]
    except json.decoder.JSONDecodeError:
        pred_dict[docids[idx]]["pred_templates"] = []

if random:
    folder_path = "predictions_BETTER/random-few/"

else:
    folder_path = "predictions_BETTER/first-few/"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
path = folder_path +str(k)+"-shot_greedy.json"
with open(path, "w") as outfile:
    json.dump(pred_dict, outfile, indent=4)
destroy_process_group()