'''
Class for prompt generation
'''
from abc import ABC, abstractmethod
import sys 
from transformers import AutoTokenizer
from init import PROMPT_FN



class Prompt(ABC):
    def __init__(self, model_name, language, dataset_name, think):
        self.language = language
        if dataset_name == "BETTER":
            self.dataset_guidelines = PROMPT_FN["BETTER_GUIDELINES"]
        else:
            self.dataset_guidelines = PROMPT_FN["MUC_GUIDELINES"]
        self.dataset = dataset_name
        self.think = think
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @abstractmethod
    def generate_prompt(self):
        pass



class LlamaR1Prompt(Prompt):
    def generate_prompt(self,data):
        prompt = []
        user_prompt = PROMPT_FN["P_U_70BR1_REASONING"].format(language=self.language, guidelines=self.dataset_guidelines, document=data["doctext"])
        prompt.append({'role': 'user', 'content': user_prompt})
        prompt_token_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True) + self.tokenizer.encode("<think>\n")
        return prompt_token_ids

class Llama3_3Prompt(Prompt):
    def generate_prompt(self,data):
        prompt = []
        system_prompt = PROMPT_FN["P_S_LLAMA_JSON"].format(language=self.language, guidelines=self.dataset_guidelines)
        prompt.append({'role': 'system', 'content': system_prompt})
        user_prompt = PROMPT_FN["P_U_LLAMA_JSON"].format(document=data["doctext"])
        prompt.append({'role': 'user', 'content': user_prompt})
        prompt_token_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
        return prompt_token_ids

class Qwen3Prompt(Prompt):
    def generate_prompt(self,data):
        prompt = []
        system_prompt = PROMPT_FN["P_S_QWEN_JSON"].format(language=self.language, guidelines=self.dataset_guidelines)
        prompt.append({'role': 'system', 'content': system_prompt})
        user_prompt = PROMPT_FN["P_U_QWEN_JSON"].format(document=data["doctext"])
        prompt.append({'role': 'user', 'content': user_prompt})
        prompt_token_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, enable_thinking=self.think, tokenize=True)
        return prompt_token_ids
class Reward(Prompt):
    def generate_prompt(self,data,template=[]):
        prompt = []
        system_prompt = PROMPT_FN["P_S_QWEN_JSON"].format(language=self.language, guidelines=self.dataset_guidelines)
        prompt.append({'role': 'system', 'content': system_prompt})
        user_prompt = PROMPT_FN["P_U_QWEN_JSON"].format(document=data["doctext"])
        prompt.append({'role': 'user', 'content': user_prompt})
        prompt.append({'role': 'assistant', 'content': template})
        #prompt_token_ids = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True, max_length=20000, return_tensors="pt").to("cuda")
        prompt_token_ids = self.tokenizer.apply_chat_template(prompt, tokenize=True)
        return prompt_token_ids


def prompt_factory(model_name, language, dataset_name, think):
    if "R1" in model_name:
        return LlamaR1Prompt(model_name, language, dataset_name, think)
    elif "Qwen3" in model_name:
        return Qwen3Prompt(model_name, language, dataset_name, think)
    elif "Reward" in model_name:
        return Reward(model_name, language, dataset_name, think)
    else:
        return Llama3_3Prompt(model_name, language, dataset_name, think)