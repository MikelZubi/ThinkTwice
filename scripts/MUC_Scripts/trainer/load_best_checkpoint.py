from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import torch
import json
import argparse
import os
import shutil
import random as rd
from torch.distributed import destroy_process_group
import csv
import argparse

#Argument parser
parser = argparse.ArgumentParser(description='Arguments required')
parser.add_argument('--r1', dest='r1', action='store_true',
                    help='Llama 8B R1 destill model')
parser.set_defaults(r1=False)


args = parser.parse_args()






# Read CSV files


if not args.r1:
    csv_files = [
        "predictions/DEV/MUC_simplified_SFT_JSON/en/results.csv",
        "predictions/DEV/MUC_simplified_SFT_Natural_Reasoning/en/results.csv",
        "predictions/DEV/MUC_simplified_SFT_Reasoning/en/results.csv",
    ]
    merge_paths = [
        "Model_JSON",
        "Model_Natural_Reasoning",
        "Model_Reasoning",
    ]
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
else:
    csv_files = [
        "predictions/DEV/MUC_simplified_SFT_JSONR1/en/results.csv",
        "predictions/DEV/MUC_simplified_SFT_Natural_ReasoningR1/en/results.csv",
        "predictions/DEV/MUC_simplified_SFT_ReasoningR1/en/results.csv",
    ]
    merge_paths = [
        "Model_JSONR1",
        "Model_Natural_ReasoningR1",
        "Model_ReasoningR1",
    ]
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype= torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)


for csv_file, merge_path in zip(csv_files, merge_paths):
    best_f1 = 0.0
    chkpt_path = None
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                f1 = float(row['iterx_muc_slot_f1'])
                if f1 > best_f1:
                    best_f1 = f1
                    chkpt_path = row['checkpoint']

    if not chkpt_path:
        raise ValueError("No checkpoint found in CSV files")
    print(merge_path)
    peft_model = PeftModel.from_pretrained(model, merge_path + "V2/" + chkpt_path)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained(merge_path,safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(merge_path)