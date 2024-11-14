"""Example of integrating `outlines` with `vllm`."""

import vllm
from pydantic import BaseModel, conset
from transformers import AutoTokenizer
from outlines.integrations.vllm import JSONLogitsProcessor
import json
import time
from transformers import set_seed
import sys

class Template(BaseModel):
    incident_type: str
    PerpInd: list[list[str]]
    PerpOrg: list[list[str]]
    Target: list[list[str]]
    Victim: list[list[str]]
    Weapon: list[list[str]]


class Base(BaseModel):
    templates: conset(item_type=Template, min_length=0, max_length=7)

def generate_prompt(doc,tokenizer, k=2):
    with open("multimuc/data/multimuc_v1.0/en/train_preprocess.jsonl") as f:
        count = 0
        prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: arson, attack, bombing, kidnapping, robbery, and forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (A inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act). To better undestand the task you will have some few-shot information'}]
        for line in f:
            template_str = line.split(', "templates": ')[1][:-2]
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
k = 45
seed = 42
set_seed(seed)
inputs = []
docids = []
pred_dict = {}

with open("multimuc/data/multimuc_v1.0/en/test.jsonl") as f:
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
        prompt = generate_prompt(data["doctext"],tokenizer,k)
        inputs.append(prompt)
denboa1 = time.time()
llm = vllm.LLM(model=model_name, tensor_parallel_size=1, enforce_eager=True)
print("Time taken: ", time.time()-denboa1)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
logits_processor = JSONLogitsProcessor(schema=Base, llm=llm, whitespace_pattern=r" ?")
print("Generating...")
result = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=vllm.SamplingParams(
        logits_processors=[logits_processor],
        temperature=0.0,
        max_tokens=1000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)
output = "output.txt"
with open(output, "w") as f:
    for output in result:
        f.write(output.outputs[0].text + "\n")
for idx, output in enumerate(result):
    print(output.outputs[0].text)
    pred_dict[docids[idx]]["pred_templates"] = json.loads(output.outputs[0].text)["templates"]

with open("predictions/"+str(k)+"-shot_greedy.json", "w") as outfile:
    json.dump(pred_dict, outfile, indent=4)