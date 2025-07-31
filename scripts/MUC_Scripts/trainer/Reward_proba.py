#import json
from create_data import create_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig #DataCollatorForPreference
from peft import LoraConfig
import torch
from datasets import load_from_disk
from accelerate import PartialState
#import numpy as np 
#from utils import *
from transformers.integrations import HfDeepSpeedConfig, deepspeed_config
import argparse
def add_margin(row):
    # Assume you have a score_chosen and score_rejected columns that you want to use to compute the margin
    return {'margin': row['score_chosen'] - row['score_rejected']}

#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the Reward trainer')
#parser.add_argument('--reasoning', dest='reasoning', action='store_true',
#                    help='Use reasoning.')
#parser.add_argument('--natural-reasoning', dest='natural_reasoning', action='store_true',
#                    help='Use natural reasoning, default to artificial.')
parser.add_argument('--batch-size', dest="batch_size", type=int)
parser.add_argument("--base-model", dest="base_model", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
parser.add_argument("--out-dir", dest="out_dir", type=str, default='Model_JSONV2')
parser.add_argument("--model-path", dest="model_path", type=str)
parser.add_argument("--lora", dest="lora", action='store_true')
parser.add_argument("--epochs", dest="epochs", type=int)




#parser.set_defaults(natural_reasoning=False)
parser.set_defaults(batch_size=2)
parser.set_defaults(lora=False)
parser.set_defaults(sampled_template=False)
parser.set_defaults(epochs=10)
args = parser.parse_args()


max_seq_length = 5000
modelname = args.base_model
model_path = args.model_path + modelname
if modelname == "DeepSeek-R1-Distill-Llama-8B":
    chat = False
else:
    chat = True

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side = 'left') #Igual ezkerrean jarri beharko da?
tokenizer.model_max_length = max_seq_length
splits = ["train"]
data = create_dataset(tokenizer,'en',chat=chat,rejectionSampling=False, Reward=True, n=-1, splits=splits)








instruct_template = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
if modelname == "DeepSeek-R1-Distill-Llama-70B":
    response_template = "<｜Assistant｜>"
elif modelname != "DeepSeek-R1-Distill-Llama-8B":
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
else:
    response_template = "<ASSISTANT>"
#data_collator = DataCollatorForPreference(pad_token_id=tokenizer.pad_token_id)


batch_size = args.batch_size
lora = args.lora


out_dir = args.out_dir
run_name = "Reward_Model_PROBATAKO_OSOA"
if lora:
    out_dir = out_dir + "_LORA"
    run_name = run_name + "_LORA"

'''
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
        asdfasdf

    # Replace -100 in the preds as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    post_preds, post_labels = postprocess(IDS,decoded_preds,decoded_labels)
    

    results = score(pred_data=post_preds, ref_data=post_labels)
    return {'precision': results["iterx_muc_slot_p"], 'recall': results["iterx_muc_slot_r"], 'f1': results["iterx_muc_slot_f1"]}
'''

data_train = data['train']

#gradient_acumulation = 128//(args.batch_size * 4)
gradient_acumulation = 1
# Define the trainer
if lora:
    peft_config = LoraConfig(
        task_type='CAUSAL_LM', inference_mode=False,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj'], r=128, lora_alpha=128
    )
    lr = 2e-4
else:
    peft_config = None
    lr = 5e-6
deepspeed = "scripts/MUC_Scripts/trainer/config/deepspeed_zero3.json"
#train_epochs = (32//n) * 4
train_epochs = args.epochs
config = RewardConfig(
    gradient_accumulation_steps=gradient_acumulation,
    output_dir=out_dir,
    run_name=run_name,
    overwrite_output_dir=True,
    #save_strategy='epoch',
    save_strategy='steps',
    save_steps=0.25,
    num_train_epochs=train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.0,
    learning_rate=lr,
    bf16=True,
    report_to='wandb', # 'wandb',
    do_train=True,
    #use_liger=True,
    max_length=max_seq_length,
    deepspeed = deepspeed,
    #packing=False,
    #label_names=["labels"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs = dict(use_reentrant=False),
    logging_steps=25
)

if deepspeed_config() is None:
    deepspeed_config_path = config.deepspeed
    if deepspeed_config_path is not None:
        deepspeed_config_obj = HfDeepSpeedConfig(deepspeed_config)
    else:
        raise ValueError("Deepspeed config is not provided.")
#device_string = PartialState().process_index
model = AutoModelForSequenceClassification.from_pretrained(model_path,  num_labels=1,torch_dtype= torch.bfloat16, attn_implementation='flash_attention_2')
#model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", torch_dtype= torch.bfloat16, attn_implementation='flash_attention_2')
model.config.pad_token_id = tokenizer.pad_token_id
model.config.pad_token = tokenizer.pad_token



train = RewardTrainer(
        model=model,
        args=config,
        train_dataset=data_train,
        processing_class=tokenizer,
        peft_config=peft_config,
        #compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        #data_collator=data_collator,
    )
# Train the model
train.train()
#model = train.model
#model = model.merge_and_unload()
#train.model = model
#train.save_model(out_dir)

