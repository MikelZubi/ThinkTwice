#import json
from create_data import create_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
import torch
from accelerate import PartialState
#import numpy as np 
#from utils import *
from transformers.integrations import HfDeepSpeedConfig, deepspeed_config
import argparse


#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the SFT trainer')
#parser.add_argument('--reasoning', dest='reasoning', action='store_true',
#                    help='Use reasoning.')
#parser.add_argument('--natural-reasoning', dest='natural_reasoning', action='store_true',
#                    help='Use natural reasoning, default to artificial.')
parser.add_argument('--sampling', dest='sampling', action='store_true')
parser.add_argument('--batch-size', dest="batch_size", type=int)
parser.add_argument("--base-model", dest="base_model", type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')
parser.add_argument("--out-dir", dest="out_dir", type=str, default='Model_JSONV2')
parser.add_argument("--model-path", dest="model_path", type=str)
parser.add_argument("--n", dest="n", type=int)

parser.set_defaults(sampling=False)
#parser.set_defaults(natural_reasoning=False)
parser.set_defaults(batch_size=2)
parser.set_defaults(n=32)

args = parser.parse_args()


max_seq_length = 7000
n = args.n
modelname = args.base_model
model_path = args.model_path + modelname
sampling = args.sampling
if modelname == "DeepSeek-R1-Distill-Llama-8B":
    chat = False
else:
    chat = True

tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='right') #Igual ezkerrean jarri beharko da?
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
print(tokenizer.pad_token_id)
print(tokenizer.pad_token)
tokenizer.model_max_length = max_seq_length
data = create_dataset(tokenizer,'en',chat=chat,rejectionSampling=sampling, n=n)





instruct_template = "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
if modelname == "DeepSeek-R1-Distill-Llama-70B":
    response_template = "<｜Assistant｜>"
elif modelname != "DeepSeek-R1-Distill-Llama-8B":
    response_template = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
else:
    response_template = "<ASSISTANT>"
#data_collator = DataCollatorForCompletionOnlyLM(response_template=tokenizer.encode(response_template,add_special_tokens=False), instruction_template=instruct_template,tokenizer=tokenizer)
data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)


batch_size = args.batch_size
if sampling:
    out_dir = args.out_dir + "Sampling_LORA_" + str(n)
    run_name = "SFT_Sampling_" + str(n)
else:
    out_dir = args.out_dir + "JSON_LORA"
    run_name = args.out_dir + "SFT_JSON"

peft_config = LoraConfig(
        task_type='CAUSAL_LM', inference_mode=False, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj','gate_proj'], r=128, lora_alpha=128
    )

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
print(data_train[0])
gradient_acumulation = 128//(args.batch_size * 4)
# Define the trainer
deepspeed = "scripts/MUC_Scripts/trainer/config/deepspeed_zero3.json"
config = SFTConfig(
    gradient_accumulation_steps=gradient_acumulation,
    output_dir=out_dir,
    run_name=run_name,
    overwrite_output_dir=True,
    save_strategy='epoch',
    num_train_epochs=4,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=5e-5,
    learning_rate=2e-4,
    bf16=True,
    report_to='none', # 'wandb',
    do_train=True,
#    use_liger=True,
    max_seq_length=max_seq_length,
    deepspeed = deepspeed,
    packing=False,
    label_names=["labels"],
    gradient_checkpointing=True,
    #logging_steps=50
)

if deepspeed_config() is None:
    deepspeed_config_path = config.deepspeed
    if deepspeed_config_path is not None:
        deepspeed_config_obj = HfDeepSpeedConfig(deepspeed_config)
    else:
        raise ValueError("Deepspeed config is not provided.")
#device_string = PartialState().process_index
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype= torch.bfloat16, attn_implementation='flash_attention_2')
#model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", torch_dtype= torch.bfloat16, attn_implementation='flash_attention_2')
model.config.pad_token_id = tokenizer.pad_token_id
model.config.pad_token = tokenizer.pad_token


train = SFTTrainer(
        model=model,
        args=config,
        train_dataset=data_train,
        processing_class=tokenizer,
        #peft_config=peft_config,
        #compute_metrics=compute_metrics,
        #preprocess_logits_for_metrics = preprocess_logits_for_metrics,
        data_collator=data_collator,
    )
# Train the model
train.train()
#model = train.model
#model = model.merge_and_unload()
#train.model = model
#train.save_model(out_dir)

