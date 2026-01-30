import json
import os 
import random
import argparse

BETTER_PATHS = ["en_string"]
MUC_PATHS = ["en", "ar", "fa", "ko", "ru", "zh"]
SUBPARTS = {"BETTER": BETTER_PATHS, "MUC": MUC_PATHS}


def find_paths(path="results/{dataset}/zeroshot/test", datasets=["BETTER", "MUC"]):
    all_paths = []
    for dataset in datasets:
        current_path = path.format(dataset=dataset)
        subpart = SUBPARTS[dataset]
        for sp in subpart:
            all_paths.append(os.path.join(current_path, sp))
    return all_paths


def boostrapp(input_path, output_path):
    with open(input_path, "r") as f:
        lines = f.readlines()
        new_data = []
        for line in lines:
            data = json.loads(line)
            chosen = random.choices(data["pred_json"], k=len(data["pred_json"]))
            new_line = {"docid": data["docid"], "doctext": data["doctext"], "pred_json": chosen}
            new_data.append(json.dumps(new_line))   
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(new_data))

def boostrapp_path(path, n=3):
    for i in range(n):
        out_path = os.path.join(path,f"boostrapp{i}")
        for file in os.listdir(path):
            _, extension = os.path.splitext(file)
            if extension == ".json":
                current_input_file = os.path.join(path, file)
                current_output_file = os.path.join(out_path, file)
                boostrapp(current_input_file, current_output_file)

                

if __name__ == "__main__":
    random.seed(42)
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--n", type=int, default=3, help="Number of boostrapp samples to create")
    args = argument_parser.parse_args()
    n = args.n
    paths = find_paths()
    for path in paths:
        boostrapp_path(path, n=n)


                