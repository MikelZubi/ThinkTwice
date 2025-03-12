from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import json
from transformers import set_seed
import os
from torch.distributed import destroy_process_group
import sys



LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_promt(data,step,tokenizer,language_code):
    language = LANGUAGE_MAP[language_code]
    prompt = [{'role': 'user', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+'. For that, you need to follow two steps. The first one is the following: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage". The second and last step is to extract the entities of the document that take part on the incident types that you have extracted in the previous step. The entities can be of the following types: "A person responsible for the incident (PerpInd)", "An organization responsible for the incident (PerpOrg)", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack (Victim)", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)". After that, I will give you the correct answer and you will need to correct your previous reasoning steps. The document is the next one: ' + data["doctext"]}]
    if step == 2:
        prompt.append({"role":"assitant","content":data["pred_reasoning"]})
        prompt.append({"role":"user", "content": "The correct answer is: " + data["reasoning"] +  "\n\n Now, knowing the correct answer, create the correct reasoning steps to obtain that answer"})
    prompt_token_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    return prompt_token_ids


file_paths = []
output_file_paths = []
#TODO: Hizkuntza guztik jarri
languages = ["en"]
map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}
split = str(sys.argv[1])
for language in languages:
    path_read = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
    path_write = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_CoT.jsonl"
    if os.path.exists(path_write):
        os.remove(path_write)
    file_paths.append(path_read)
    output_file_paths.append(path_write)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
set_seed(42)
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, tensor_parallel_size=2, gpu_memory_utilization=0.95, max_model_len=40000)
inputs = []
pre_dicts = []


for i in range(len(file_paths)):

    #STEP 1: Created the predicted reasoning

    with open(file_paths[i], 'r') as file:
        for line in file:
            data = json.loads(line)
            pre_dict = data
            inputs.append(generate_promt(pre_dict,1,tokenizer,languages[i]))
            pre_dicts.append(pre_dict)


    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    result_1 = llm.generate(
        prompt_token_ids=inputs,
        sampling_params=SamplingParams(
            temperature=0.6, #Recommended value
            max_tokens=4000,
            stop_token_ids=terminators,
        ),
        use_tqdm=True
    )

    for idx, output in enumerate(result_1):
        pre_dicts[idx]["pred_reasoning"] = output.outputs[0].text





    #STEP 2: Created the corrected reasoning
    inputs.clear()
    for i in range(len(file_paths)):
        for pre_dict in pre_dicts:
                inputs.append(generate_promt(pre_dict,2,tokenizer,languages[i]))


    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")]  
    result_2 = llm.generate(
        prompt_token_ids=inputs,
        sampling_params=SamplingParams(
            temperature=0.6, #Recommended value
            max_tokens=4000,
            stop_token_ids=terminators,
        ),
        use_tqdm=True
    )

    for idx, output in enumerate(result_2):
        pre_dicts[idx]["corrected_reasoning"] = "<think>\n " + output.outputs[0].text.split("</think>")[0] + "</think>"
        print(pre_dicts[idx]["corrected_reasoning"])


    with open(output_file_paths[i], 'a') as output_file:
        for line in pre_dicts:
            output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



