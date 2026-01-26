from create_data import create_dataset
from transformers import AutoTokenizer
from datasets import load_from_disk


max_seq_length = 5500
tokenizer = AutoTokenizer.from_pretrained("/scratch/ehu_p518_1/ehu_p518_1_1/Ereduak/Qwen3-Embedding-8B", padding_side = 'left') #Igual ezkerrean jarri beharko da?
#tokenizer.pad_token = "<|finetune_right_pad_id|>"
#tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
tokenizer.model_max_length = max_seq_length

splits = ["train"]
data = create_dataset(tokenizer,'en',modelname="Qwen3-Embedding-8B",chat=True,rejectionSampling=False, Reward=True, n=-1, splits=splits, tokenize=True, guidelines=True)
data.save_to_disk("/scratch/ehu_p518_1/ehu_p518_1_1/data/reward")