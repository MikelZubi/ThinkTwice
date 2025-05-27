from huggingface_hub import snapshot_download

snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", local_dir ="/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-70B")
snapshot_download("meta-llama/Meta-Llama-3.1-8B-Instruct", local_dir ="/leonardo_work/EUHPC_E04_042/BaseModels/Meta-Llama-3.1-8B-Instruct")
snapshot_download("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", local_dir ="/leonardo_work/EUHPC_E04_042/BaseModels/DeepSeek-R1-Distill-Llama-8B")