#from create_data import create_dataset
#from transformers import AutoTokenizer
import sys 
sys.path.append("class_data")
from MUC_Class_simplified import *
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from vllm import LLM, SamplingParams
from create_data import create_dataset

modelname= "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(modelname)
#data = create_dataset(tokenizer, 'en', r1=True, reasoning=False, natural_reasoning=False)
#print(data["train"]["text"][0])
#print(Base.model_json_schema())

llm = LLM(model=modelname, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.8)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    tokenizer.convert_tokens_to_ids("</think>")]  
print("Generating...")
data = create_dataset(tokenizer, 'en', r1=True, reasoning=True, natural_reasoning=True)
inp = data['train']['text'][0].split("<ASSISTANT>")[0]
print(inp)
inputs = tokenizer(data['train']['text'][0].split("<ASSISTANT>")[0] + "<think>\n")['input_ids']
result = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        #guided_decoding=guided_decoding_params,
        temperature=0.0,
        max_tokens=4000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)
print(tokenizer.encode("</think>"))
print(result)
exit()
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),]
print("Generating...")
result = llm.generate(
    prompt_token_ids=inputs,
    sampling_params=SamplingParams(
        #guided_decoding=guided_decoding_params,
        temperature=0.0,
        max_tokens=4000,
        stop_token_ids=terminators,
    ),
    use_tqdm=True
)
print(result)
