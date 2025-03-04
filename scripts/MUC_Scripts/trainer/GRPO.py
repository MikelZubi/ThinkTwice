from create_data import create_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig
import torch
from accelerate import PartialState
#from utils import *
import argparse
import requests




#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the SFT trainer')
parser.add_argument('--reasoning', dest='reasoning', action='store_true',
                    help='Use reasoning.')
parser.add_argument('--natural-reasoning', dest='natural_reasoning', action='store_true',
                    help='Use natural reasoning, default to artificial.')
parser.add_argument('--model-path', dest="model_path", type=str)
parser.add_argument('--batch-size', dest="batch_size", type=int)
parser.add_argument('--gradient-accumulation-steps', dest="gradient_accumulation_steps", type=int)
parser.add_argument('--distributed', dest='distributed', action='store_true',
                    help='Use distributed training.')
parser.add_argument('--r1', dest='r1', action='store_true',
                    help='Llama 8B R1 destill model')
parser.set_defaults(r1=False)
parser.set_defaults(distributed=False)
parser.set_defaults(reasoning=False)
parser.set_defaults(natural_reasoning=False)
parser.set_defaults(model_path='meta-llama/Meta-Llama-3.1-8B-Instruct')
parser.set_defaults(batch_size=2)

args = parser.parse_args()




max_seq_length = 2200
modelname = args.model_path
print(modelname)
tokenizer = AutoTokenizer.from_pretrained(modelname, padding_side='right') #Igual ezkerrean jarri beharko da?
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.pad_token_id = tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0]
tokenizer.model_max_length = max_seq_length
data = create_dataset(tokenizer,'en',r1=args.r1,reasoning=args.reasoning,natural_reasoning=args.natural_reasoning, GRPO=True)


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

SERVER = True
batch_size = args.batch_size
gradient_accumulation_steps = args.gradient_accumulation_steps
if args.reasoning and not args.natural_reasoning: 
    max_seq_length = 5000
    out_dir = "Model_GRPO_Reasoning"
    run_name = 'GRPO_Reasoning'
    if args.r1:
        out_dir = out_dir + "R1"
        run_name = run_name + "R1"
    deepspeed = "scripts/MUC_Scripts/trainer/config/deepspeed_zero2_reasoning.json"
    REASONING = True
elif args.reasoning and args.natural_reasoning:
    max_seq_length = 5000
    out_dir = "Model_GRPO_Natural_Reasoning"
    run_name = 'GRPO_Natural_Reasoning'
    if args.r1:
        out_dir = out_dir + "R1"
        run_name = run_name + "R1"
    deepspeed = "scripts/MUC_Scripts/trainer/config/deepspeed_zero2_reasoning.json"
    REASONING = True
else:
    out_dir = "Model_GRPO_JSON"
    run_name = 'GRPO_JSON'
    if args.r1:
        out_dir = out_dir + "R1"
        run_name = run_name + "R1"
    deepspeed = "scripts/MUC_Scripts/trainer/config/deepspeed_zero2.json"
    REASONING = False
peft_config = LoraConfig(
        task_type='CAUSAL_LM', inference_mode=False, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj','gate_proj'], r=128, lora_alpha=128
    )




def compute_server_rewards(completions, **kwargs):
    response = requests.post('http://localhost:4416/reward',json={'completions': completions, 'ground_truths': kwargs['ground_truth'], "reasoning": REASONING})
    reward = response.json()
    return reward


data_train = data['train']
data_dev = data['dev']
# Define the trainer

config = GRPOConfig(
    output_dir=out_dir,
    gradient_accumulation_steps=gradient_accumulation_steps,
    run_name=run_name,
    overwrite_output_dir=True,
    save_strategy='epoch',
    save_total_limit=1,
    num_train_epochs=40, 
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=5e-5,
    learning_rate=1e-5,
    beta=0.0,
    seed=42,
    bf16=True,
    report_to='wandb', # 'wandb',
    do_train=True,
    max_prompt_length=max_seq_length,
    deepspeed=deepspeed,
    evaluation_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model="reward"
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

