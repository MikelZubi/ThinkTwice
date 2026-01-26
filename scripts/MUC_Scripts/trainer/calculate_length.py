from create_data import create_dataset
from transformers import AutoTokenizer

def calculate_max_length(reasoning=False, natural_reasoning=False, split_text=False):
    
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    data = create_dataset(tokenizer,'en',reasoning=reasoning,natural_reasoning=natural_reasoning)
    train_data = data['train']['text']
    max_length = 0
    max_text = ""
    for train_text in train_data:
        if split_text:
            train_text = train_text.split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")[0]
        if len(tokenizer.encode(train_text)) > max_length:
            max_length = len(tokenizer.encode(train_text))
            max_text = train_text
    print(f"Max text: {max_text}")
    
    return max_length

if __name__ == "__main__":
    max_length = calculate_max_length(reasoning=False, natural_reasoning=False)
    print(f"Maximum sequence length: {max_length}")
    max_length = calculate_max_length(reasoning=True, natural_reasoning=False)
    print(f"Maximum sequence length with reasoning: {max_length}")
    max_length = calculate_max_length(reasoning=True, natural_reasoning=True)
    print(f"Maximum sequence length with natural reasoning: {max_length}")
    max_length = calculate_max_length(reasoning=False, natural_reasoning=True, split_text=True)
    print(f"Maximum sequence length with natural reasoning and split: {max_length}")