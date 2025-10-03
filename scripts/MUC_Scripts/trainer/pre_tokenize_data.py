from create_data import create_dataset
from transformers import AutoTokenizer
from datasets import load_from_disk
import argparse

parser = argparse.ArgumentParser(description='Arguments for the creation of the train dataset')
parser.add_argument("--modelname", dest="modelname", type=str)
parser.add_argument("--modelpath", dest="modelpath", type=str)
parser.add_argument("--guidelines", dest="guidelines", action='store_true')
parser.set_defaults(modelpath='/leonardo_work/EUHPC_E04_042/BaseModels/')
parser.set_defaults(modelname='Llama-3.1-8B-Instruct')
parser.set_defaults(guidelines=False)


modelname = parser.parse_args().modelname
modelpath = parser.parse_args().modelpath
guidelines = parser.parse_args().guidelines
max_seq_length = 5000
tokenizer = AutoTokenizer.from_pretrained(modelpath + modelname, padding_side = 'left') #Igual ezkerrean jarri beharko da?
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
tokenizer.model_max_length = max_seq_length

splits = ["train"]
if modelname == "ModernBERT-large":
    chat = False
else:
    chat = True

data = create_dataset(tokenizer,'en',chat=chat,rejectionSampling=False, Reward=True, n=-1, splits=splits, tokenize=True, guidelines=guidelines)
print("Saving dataset...")
data.save_to_disk("/leonardo_scratch/large/userexternal/mzubilla/data/reward")
print("Dataset saved.")