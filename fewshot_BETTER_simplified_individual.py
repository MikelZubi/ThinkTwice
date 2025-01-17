
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
import time
from transformers import set_seed
import random as rd
import os
from torch.distributed import destroy_process_group
from BETTER_Granular_Class_simplified import *




def generate_prompt(doc,tokenizer,template_class,k=2, random=True):
    with open("phase2/phase2.granular.eng.preprocess-train-simplified.jsonl") as f:
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
                inputs = json.loads(line)
                templates = []
                for template in inputs["templates"]:
                    if template["template_type"]== template_class:
                        templates.append(template)
                template_dict = {"templates":templates}
                template_str = json.dumps(template_dict)
                prompt.append({"role":"user","content":inputs["doctext"]})
                prompt.append({"role":"assistant","content":template_str})
                count += 1
                if count >= k:
                    break
        prompt.append({"role":"user","content":doc})
        prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
        return prompt_token_ids
    
class_dict = {"Protestplate": Indv_Protestplate, "Corruplate": Indv_Corruplate, "Terrorplate": Indv_Terrorplate, "Epidemiplate": Indv_Epidemiplate, "Disasterplate": Indv_Disasterplate, "Displacementplate": Indv_Displacementplate}

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
#k = int(sys.argv[1])
#if sys.argv[2] == "R":
#    random = True
#else:
#    random = False
#language = str(sys.argv[3])
seed = 42
set_seed(seed)
rd.seed(seed)
denboa1 = time.time()
llm = LLM(model=model_name, tensor_parallel_size=1, enforce_eager=True)
print("Time taken: ", time.time()-denboa1)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
guided_decoding_params_list = []
for class_key in class_dict:
    template_class = class_dict[class_key]
    guided_decoding_params = GuidedDecodingParams(json=template_class.model_json_schema())
    guided_decoding_params_list.append(guided_decoding_params)
    print("Time taken: ", time.time()-denboa1)
for random in [True,False]:
    for k in range(0,61):
        pred_all = []
        for guided_decoding_params, class_key in zip(guided_decoding_params_list, class_dict):
            inputs = []
            with open("phase2/phase2.granular.eng.preprocess-dev-simplified.jsonl") as f:
                for line in f:
                    pred_dict = {}
                    data = json.loads(line)
                    if class_key == "Protestplate":
                        docid = data["docid"]
                        pred_dict["docid"] = docid
                        pred_dict["doctext"] = data["doctext"]
                        pred_all.append(pred_dict)
                    #pred_dict[docid]["gold_templates"] = data["templates"]
                    prompt = generate_prompt(data["doctext"],tokenizer,class_key,k,random)
                    inputs.append(prompt)
            result = llm.generate(
                prompt_token_ids=inputs,
                sampling_params=SamplingParams(
                    guided_decoding=guided_decoding_params,
                    temperature=0.0,
                    max_tokens=4000,
                    stop_token_ids=terminators,
                ),
                use_tqdm=True
            )
            output = "output.txt"
            with open(output, "w") as f:
                for output in result:
                    f.write(output.outputs[0].text + "\n")
            if class_key == "Protestplate":
                for i in range(len(pred_all)):
                    pred_all[i]["templates"] = []
            for idx, output in enumerate(result):
                try:
                    pred_all[idx]["templates"] += json.loads(output.outputs[0].text)["templates"]
                except json.decoder.JSONDecodeError:
                    pred_all[idx]["templates"] += []
            
            if class_key == "Displacementplate":

                if random:
                    folder_path = "predictions_BETTER_simplified_individuals/random-few/"

                else:
                    folder_path = "predictions_BETTER_simplified_individuals/first-few/"

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                path = folder_path +str(k)+"-shot_greedy.jsonl"
                with open(path, "w") as outfile:
                    for value in pred_all:
                        json.dump(value, outfile, ensure_ascii=False)
                        outfile.write("\n")

destroy_process_group()