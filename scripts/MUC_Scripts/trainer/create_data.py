from datasets import Dataset, DatasetDict
from collections import defaultdict
import json
import copy as cp


def generate_prompt_train(tokenizer, language,line_dict,reasoning_tag,reasoning=False):
    prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act).'}]
    prompt.append({"role": "user", "content": line_dict["doctext"]})
    template_str = json.dumps(line_dict["templates"], ensure_ascii=False)
    if not reasoning:
        prompt.append({"role": "assistant", "content": template_str})
    else:
        prompt.append({"role": "assistant", "content": line_dict[reasoning_tag] + "\n\n JSON: \n" + template_str})
    tokenized_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    grpo_prompt = tokenizer.apply_chat_template(prompt[0:2], tokenize=False, add_generation_prompt=True)
    grpo_completion = tokenizer.apply_chat_template(prompt[2:], tokenize=False, add_generation_prompt=False)
    return tokenized_prompt, grpo_prompt, grpo_completion


def generate_text_train(language,line_dict,reasoning_tag,reasoning=False):
    prompt = '<SYSTEM> You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act).\n\n'
    prompt += '<USER> '+ line_dict["doctext"] + "\n\n"
    grpo_prompt = cp.deepcopy(prompt)
    template_str = json.dumps(line_dict["templates"], ensure_ascii=False)
    if not reasoning:
        grpo_completion = "<ASSISTANT> " + template_str
        prompt += grpo_completion
        
    else:
        grpo_completion = "<ASSISTANT> " + line_dict[reasoning_tag] + "\n\n JSON: \n" + template_str
        prompt += grpo_completion
    return prompt, grpo_prompt, grpo_completion
    
def convert_docid(docid: str) -> str:
    return str(int(docid.split("-")[0][-1]) * 10000 + int(docid.split("-")[-1]))


def create_dataset(tokenizer, language, r1=False,reasoning=False,natural_reasoning=False, GRPO=False):

    datasetdict = {}
    for split in ["dev","train"]:
        path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/"+split+"_simplified_preprocess.jsonl"
        reasoning_tag = "reasoning"
        path_ground_truth = "multimuc/data/multimuc_v1.0/corrected/"+language+"/"+split+".jsonl"
        if natural_reasoning:
            path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/"+split+"_simplified_CoT.jsonl"
            reasoning_tag = "corrected_reasoning"

    
        data_dict = defaultdict(lambda: [])

        # Load the dataset
        with open(path_read, "r") as file:
            all_lines = file.readlines()
        with open(path_ground_truth, "r") as file:
            all_lines_gt = file.readlines()
        for line, line_gt in zip(all_lines, all_lines_gt):
                line_dict = json.loads(line)
                #id = convert_docid(line_dict["docid"]) if split == "dev" else line_dict["docid"]
                id = line_dict["docid"]
                data_dict["id"].append(id)
                #TODO
                if not r1: 
                    prompt, init_prompt, completion = generate_prompt_train(tokenizer, language,line_dict,reasoning_tag=reasoning_tag,reasoning=reasoning)
                else:
                    prompt, init_prompt, completion = generate_text_train(language,line_dict,reasoning_tag=reasoning_tag,reasoning=reasoning)
                if not GRPO:
                    data_dict["text"].append(prompt)
                if GRPO: 
                    data_dict["prompt"].append(init_prompt)
                    data_dict["completion"].append(completion)
                    line_dict_gt = json.loads(line_gt)
                    data_dict["ground_truth"].append(json.dumps(line_dict_gt["templates"], ensure_ascii=False))

                #data_dict["messages"].append(prompt)
        # Create a Dataset object
        dataset = Dataset.from_dict(data_dict)
        datasetdict[split] = dataset
    return DatasetDict(datasetdict)