from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from vllm import LLM, SamplingParams
import torch
import argparse

#Argument parser
parser = argparse.ArgumentParser(description='Arguments required')
parser.add_argument('--model-name',dest="model_name", type=str, help='Model name')
parser.add_argument('--chkpt-path',dest="chkpt_path", type=str, help='Checkpoint path')
parser.add_argument('--out-dir',dest="out_dir", type=str, help='Output directory')

parser.set_defaults(model_name="/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B")
parser.set_defaults(out_dir="/leonardo_scratch/large/userexternal/mzubilla/DPO/Merged_Model")

args = parser.parse_args()
model_name = args.model_name
chkpt_path = args.chkpt_path

merge_path = args.out_dir
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype= torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Path to save the merged model
# Merge LoRA and base model and save
peft_model = PeftModel.from_pretrained(model, chkpt_path)
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained(merge_path,safe_serialization=True)
tokenizer.save_pretrained(merge_path)