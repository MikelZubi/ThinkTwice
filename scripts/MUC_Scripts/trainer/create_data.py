from datasets import Dataset, DatasetDict
from collections import defaultdict
import json
import copy as cp
import sys
sys.path.append("prompt_library")
from init import PROMPT_FN



LANGUAGE_MAP = {"en": "English", "ar": "Arabic", "fa": "Farsi", "ko": "Korean", "ru": "Russian", "zh": "Chinese"}


def generate_prompt_train(tokenizer, language_code,line_dict,tag,reasoning=False,obtain_reasoning=False):
    language = LANGUAGE_MAP[language_code]
    if reasoning:
        if obtain_reasoning:
            prompt = [{'role': 'system', 'content': PROMPT_FN["P_U_MUC_70BR1_OBTAIN_REASONING"].format(language=language, document=line_dict["doctext"], template=line_dict["template"])}]
            prompt.append({"role": "assistant", "content": line_dict["reasoning"]})
        else:
            prompt = [{'role': 'user', 'content': PROMPT_FN["P_U_MUC_70BR1_REASONING"].format(language=language, document=line_dict["doctext"])}]
            prompt.append({"role": "assistant", "content": line_dict[tag]})
    else:
        prompt = [{'role': 'system', 'content': PROMPT_FN["P_S_MUC_LLAMA_JSON"].format(language=language)}]
        prompt.append({"role": "user", "content": PROMPT_FN["P_U_MUC_LLAMA_JSON"].format(document=line_dict["doctext"])})
        template_str = json.dumps(line_dict[tag], ensure_ascii=False)
        prompt.append({"role": "assistant", "content": template_str})
    chat_prompt = tokenizer.apply_chat_template(prompt, add_generation_prompt=False, tokenize=False)
    if reasoning:
        corrected_prompt = chat_prompt.replace("</THINK_TOKENA>", "</think>")
    else:
        corrected_prompt = chat_prompt
    return corrected_prompt


def generate_text_train(language_code,line_dict,tag,reasoning=False,cold_start=False):
    language = LANGUAGE_MAP[language_code]
    prompt = ""
    if reasoning:
        prompt = PROMPT_FN["P_U_MUC_8BR1_REASONING"].format(language=language, document=line_dict["doctext"])
        prompt += PROMPT_FN["P_A_MUC_8BR1_REASONING"].format(reasoning=line_dict[tag])
    
    return prompt
'''
def generate_prompt_train_old(tokenizer, language_code,line_dict,tag,reasoning=False,cold_start=False):
    language = LANGUAGE_MAP[language_code]
    if reasoning:
        prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+'. For that, you need to follow two steps. The first one is the following: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage". The second and last step is to extract the entities of the document that take part on the incident types that you have extracted in the previous step. The entities can be of the following types: "A person responsible for the incident (PerpInd)", "An organization responsible for the incident (PerpOrg)", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack (Victim)", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)". After that, you need to create a JSON that stores the information that you have extracted.'}]
    else:
        prompt = [{'role': 'system', 'content': 'You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act).'}]
    prompt.append({"role": "user", "content": '"' + line_dict["doctext"] + '"'})
    template_str = json.dumps(line_dict["templates"], ensure_ascii=False)
    if not reasoning:
        prompt.append({"role": "assistant", "content": template_str})
    else:
        if cold_start:
            prompt.append({"role": "assistant", "content": line_dict[tag]})
        else:
            prompt.append({"role": "assistant", "content": line_dict[tag] + "\n JSON: \n" + template_str})
    tokenized_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
    grpo_prompt = tokenizer.apply_chat_template(prompt[0:2], tokenize=False, add_generation_prompt=True)
    grpo_completion = tokenizer.apply_chat_template(prompt[2:], tokenize=False, add_generation_prompt=False)
    return tokenized_prompt, grpo_prompt, grpo_completion


def generate_text_train_old(language_code,line_dict,tag,reasoning=False,cold_start=False):
    language = LANGUAGE_MAP[language_code]
    if reasoning:
        prompt = '<USER> You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+'. For that, you need to follow two steps. The first one is the following: you need to identify what incident types are happening in the next document. The incident types can be the following ones: "kidnapping", "attack", "bombing", "robbery", "arson", and "forced work stoppage". The second and last step is to extract the entities of the document that take part on the incident types that you have extracted in the previous step. The entities can be of the following types: "A person responsible for the incident (PerpInd)", "An organization responsible for the incident (PerpOrg)", "An inanimate object that was attacked (Target)", "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack (Victim)", and "A device used by the perpetrator(s) in carrying out the terrorist act (Weapon)". After that, you need to create a JSON that stores the information that you have extracted. The document is the next one:\n"' + line_dict["doctext"] +'" </USER>\n'
    else:
        prompt = '<USER> You are an expert in information extraction, you need to extract the information of the document that is provided in '+language+' as a template in JSON format. For that, first, you need to indicate what is the "incident_type", which can be: kidnapping, attack, bombing, robbery, arson, or forced work stoppage. Then, you need to fill the next slots (or leave them empty): "PerpInd" (A person responsible for the incident.), "PerpOrg" (An organization responsible for the incident.), "Target" (An inanimate object that was attacked), "Victim" (The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack), and "Weapon" (A device used by the perpetrator/s in carrying out the terrorist act). The document is the next one:\n"' + line_dict["doctext"] + '" </USER>\n'
    grpo_prompt = cp.deepcopy(prompt) + "<think>\n"
    template_str = json.dumps(line_dict["templates"], ensure_ascii=False)
    if not reasoning:
        grpo_completion = "<ASSISTANT> " + template_str + " </ASSISTANT>"
        prompt += grpo_completion
        
    else:
        if cold_start:
            grpo_completion = "<ASSISTANT> " + line_dict[tag] + " </ASSISTANT>"
        else:
            grpo_completion = "<ASSISTANT> " + line_dict[tag] + "\n JSON: \n" + template_str + " </ASSISTANT>"
        prompt += grpo_completion
    return prompt, grpo_prompt, grpo_completion
    
def convert_docid(docid: str) -> str:
    return str(int(docid.split("-")[0][-1]) * 10000 + int(docid.split("-")[-1]))
'''

def create_dataset(tokenizer, language, chat=True,DPO=False, Reward=False, GRPO=False,cold_start=False, rejectionSampling=False, n=32,splits=["train"]):

    datasetdict = {}
    for split in splits: #TODO: Buelta bat eman honi kode honek funtzionatzen du biño ya ez du zentzue ola ittea
        if rejectionSampling or DPO or Reward or GRPO:
            if DPO:
                path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/rejectionSampling/DPO.jsonl"
            elif Reward:
                path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/rejectionSampling/reward.jsonl"
            else:
                path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/rejectionSampling/train_best"+str(n)+".jsonl"
        else:
            path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/"+split+"_simplified_preprocess.jsonl"
        path_ground_truth = "multimuc/data/multimuc_v1.0/corrected/"+language+"/train.jsonl"
        #if natural_reasoning:
        #    path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/"+split+"_simplified_CoT.jsonl"


        data_dict = defaultdict(lambda: [])

        # Load the dataset
        with open(path_read, "r") as file:
            all_lines = file.readlines()
            if not GRPO:
                all_lines_gt = all_lines
        if GRPO:  #TODO: Errebisatu
            with open(path_ground_truth, "r") as file:
                all_lines_gt = file.readlines() * n
                
        for line, line_gt in zip(all_lines, all_lines_gt):
                line_dict = json.loads(line)
                #id = convert_docid(line_dict["docid"]) if split == "dev" else line_dict["docid"]
                id = line_dict["docid"]
                data_dict["id"].append(id)
                #if chat:
                    #prompt, init_prompt, completion = generate_prompt_train(tokenizer, language,line_dict,tag=tag,reasoning=reasoning,cold_start=cold_start)
                if not rejectionSampling and not DPO and not Reward and not GRPO:
                    prompt = generate_prompt_train(tokenizer, language, line_dict, tag="templates", reasoning=False, cold_start=cold_start)
                    data_dict["text"].append(prompt)



                elif rejectionSampling and not DPO and not Reward:
                    prompt = generate_prompt_train(tokenizer, language, line_dict,
                                                                        tag="completion",
                                                                        reasoning=rejectionSampling, cold_start=cold_start)
                    data_dict["text"].append(prompt)
                
                elif DPO and not Reward:
                    prompt_c = generate_prompt_train(tokenizer, language, line_dict,
                                                                        tag="chosen",
                                                                        reasoning=rejectionSampling, cold_start=cold_start)
                    data_dict["chosen"].append(prompt_c)
                    prompt_r = generate_prompt_train(tokenizer, language, line_dict,
                                                                        tag="rejected",
                                                                        reasoning=rejectionSampling, cold_start=cold_start)
                    data_dict["rejected"].append(prompt_r)
                    
                elif Reward:
                    prompt_c = generate_prompt_train(tokenizer, language, line_dict,
                                                                        tag="chosen",
                                                                        reasoning=rejectionSampling, cold_start=cold_start)
                    data_dict["chosen"].append(prompt_c)
                    prompt_r = generate_prompt_train(tokenizer, language, line_dict,
                                                                        tag="rejected",
                                                                        reasoning=rejectionSampling, cold_start=cold_start)
                    data_dict["rejected"].append(prompt_r)
                    data_dict["margin"].append(line_dict["margin"])
                
                    
                elif GRPO:
                    #data_dict["prompt"].append(init_prompt)
                    #data_dict["completion"].append(completion)
                    line_dict_gt = json.loads(line_gt)
                    data_dict["ground_truth"].append(json.dumps(line_dict_gt["templates"], ensure_ascii=False))

                #data_dict["messages"].append(prompt)
        # Create a Dataset object
        dataset = Dataset.from_dict(data_dict)
        dataset_suffle = dataset.shuffle(seed=42)
        datasetdict["train"] = dataset_suffle
    return DatasetDict(datasetdict)