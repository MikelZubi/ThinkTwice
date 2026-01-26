from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
print(tokenizer.convert_tokens_to_ids("</think>"))
#print(tokenizer.convert_tokens_to_ids("</ASSISTANT>"))
#print(tokenizer.convert_tokens_to_ids(" </ASSISTANT>"))