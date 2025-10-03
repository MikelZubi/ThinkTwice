import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, default=".jsonl")
args = parser.parse_args()

path = args.path
outputs = []
n = 1  # Number of templates to select
obtain_reasoning = True  # Flag to indicate if reasoning should be obtained
language = "en"  # Default language, can be modified as needed
for line in open(path, "r"):
    data = json.loads(line)
    id = data["docid"]
    document = data["doctext"]
    gold = data["gold_templates"]

    pred_reasoning = data["pred_reasoning"]

    outputs.append({"docid": id, "reasoning": "<think>\n" + pred_reasoning + "</THINK_TOKENA>", "template": gold, "doctext": document})

out_path = "multimuc/data/multimuc_v1.0/corrected/"+language+"/rejectionSampling/train_best"+str(n)+".jsonl"
with open(out_path, "w") as f:
    for output in outputs:
        f.write(json.dumps(output) + "\n")
