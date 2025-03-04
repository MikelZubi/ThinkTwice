from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from vllm import LLM, SamplingParams
import sys 
import torch
import json
import argparse
import os
import shutil
import random as rd



model_name = "Model_Natural_Reasoning"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype= torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
peft_model = PeftModel.from_pretrained(model, "Model_GRPO_Natural_Reasoning/checkpoint-2771")
merged_model = peft_model.merge_and_unload()
merge_path = "Model_GRPO_Natural_Reasoning"
merged_model.save_pretrained(merge_path,safe_serialization=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(merge_path)