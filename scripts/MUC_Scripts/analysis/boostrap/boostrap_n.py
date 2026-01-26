import json
import os

boostrap_sizes = [2, 4, 8, 16, 32]

path = "results/MUC/zeroshot/test/boostrap_en"
read_think = "results/MUC/zeroshot/test/boostrap_en/Qwen3-32B_think_64.jsonl"

with open(read_think, 'r') as f:
    for size in boostrap_sizes:
        output_file = f"{path}/Qwen3-32B_think_{size}.jsonl"
        if os.path.exists(output_file):
            os.remove(output_file)
    for line in f:
        for size in boostrap_sizes:
            data = json.loads(line)
            docid = data["docid"]
            pred_templates = data["pred_json"]
            doctext = data["doctext"]
            templates = data["templates"]
            new_pred_templates = pred_templates[:size]
            output_data = {
                "docid": docid,
                "pred_json": new_pred_templates,
                "doctext": doctext,
                "templates": templates
            }
            output_file = f"{path}/Qwen3-32B_think_{size}.jsonl"
            with open(output_file, 'a') as out_f:
                out_f.write(json.dumps(output_data) + '\n')

read_nothink = "results/MUC/zeroshot/test/boostrap_en/Qwen3-32B_nothink_64.jsonl"
with open(read_nothink, 'r') as f:
    for size in boostrap_sizes:
        output_file = f"{path}/Qwen3-32B_nothink_{size}.jsonl"
        if os.path.exists(output_file):
            os.remove(output_file)
    for line in f:
        for size in boostrap_sizes:
            data = json.loads(line)
            docid = data["docid"]
            pred_templates = data["pred_json"]
            doctext = data["doctext"]
            templates = data["templates"]
            new_pred_templates = pred_templates[:size]
            output_data = {
                "docid": docid,
                "pred_json": new_pred_templates,
                "doctext": doctext,
                "templates": templates
            }
            output_file = f"{path}/Qwen3-32B_nothink_{size}.jsonl"
            with open(output_file, 'a') as out_f:
                out_f.write(json.dumps(output_data) + '\n')