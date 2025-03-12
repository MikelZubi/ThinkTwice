from create_data import create_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
import torch
from accelerate import PartialState
#from utils import *
import argparse
import requests

import sys 
sys.path.append("class_data")
from MUC_Class_simplified import *





#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the SFT trainer')

parser.add_argument('--model-path', dest="model_path", type=str)
parser.add_argument('--batch-size', dest="batch_size", type=int)
parser.add_argument('--gradient-accumulation-steps', dest="gradient_accumulation_steps", type=int)
parser.add_argument('--distributed', dest='distributed', action='store_true',
                    help='Use distributed training.')
parser.add_argument('--r1', dest='r1', action='store_true',
                    help='Llama 8B R1 destill model')
parser.set_defaults(r1=True)
parser.set_defaults(distributed=True)
parser.set_defaults(model_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
parser.set_defaults(batch_size=1)

args = parser.parse_args()




max_seq_length = 2150
modelname = args.model_path
print(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname) #Igual ezkerrean jarri beharko da?
#tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='right') #Igual ezkerrean jarri beharko da?
#tokenizer.pad_token = "<|finetune_right_pad_id|>"
#tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
#tokenizer.model_max_length = max_seq_length
data = create_dataset(tokenizer,'en',r1=args.r1,reasoning=True,natural_reasoning=True, GRPO=True)


IDS = data['dev']['id']

if args.distributed:
    DIST = True
    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(modelname, device_map={'': device_string}, torch_dtype= torch.bfloat16, attn_implementation='flash_attention_2')
else:
    DIST = False
    model = AutoModelForCausalLM.from_pretrained(modelname, device_map="auto", torch_dtype= torch.bfloat16, attn_implementation='flash_attention_2')
model.config.pad_token_id = tokenizer.pad_token_id
model.config.pad_token = tokenizer.pad_token


REASONING = True
SERVER = True
deepspeed = "scripts/MUC_Scripts/trainer/config/deepspeed_zero2_scratch.json"
batch_size = args.batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
out_dir = "Model_GRPO_Scratch"
run_name = 'GRPO_Scratch'
peft_config = LoraConfig(
        task_type='CAUSAL_LM', inference_mode=False, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj','gate_proj'], r=128, lora_alpha=128
    )

terminators=[
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("</think>")
]


def compute_server_rewards(completions, **kwargs):
    response = requests.post('http://localhost:4416/reward',json={'completions': completions, 'ground_truths': kwargs['ground_truth'], "reasoning": False})
    reward = response.json()
    print(reward)
    return reward


data_train = data['train']
data_dev = data['dev']
# Define the trainer

config = GRPOConfig(
    output_dir=out_dir,
    gradient_accumulation_steps=gradient_accumulation_steps,
    run_name=run_name,
    overwrite_output_dir=True,
    save_strategy='steps',
    save_steps=0.05,
    num_train_epochs=40, 
    num_generations=6,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=5e-5,
    learning_rate=5e-6,
    num_iterations=3,
    temperature=0.7,
    beta=0.0,
    seed=42,
    bf16=True,
    report_to='wandb', # 'wandb',
    do_train=True,
    max_prompt_length=max_seq_length,
    max_completion_length=4000,
    deepspeed=deepspeed,
    evaluation_strategy='steps',
    eval_steps=0.05,
    logging_strategy='steps',
    logging_steps=0.05,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    use_vllm=True,
    vllm_device="auto",
    vllm_gpu_memory_utilization=0.65,
    vllm_max_model_len=max_seq_length,
    #vllm_guided_decoding_regex="<think>\n([\s\S]*?)\n</think>",
    vllm_guided_decoding_json=Base.model_json_schema(),
    vllm_combine_guided_decoding=True,
    vllm_stop_token_ids=terminators,
)
train = GRPOTrainer(
        model=model,
        args=config,
        processing_class=tokenizer,
        train_dataset=data_train,
        eval_dataset=data_dev,
        peft_config=peft_config,
        reward_funcs=compute_server_rewards,
    )
# Train the model
train.train()
model = train.model
model = model.merge_and_unload()
train.model = model
train.save_model(out_dir)

