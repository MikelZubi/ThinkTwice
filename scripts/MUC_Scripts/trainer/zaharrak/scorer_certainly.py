import argparse
import json
import numpy as np


def selected_max_certainly(data):
    template = ["ERROR"]
    while template == ["ERROR"] or template == [["ERROR"]]:
        idx = np.argmax(data["selected_mean"])
        template = data["pred_json"][idx]
        data["selected_mean"][idx] = -1  # Mark as used

    return template


#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--read', dest='read', type=str)
parser.add_argument("--out", dest="out", type=str)

args = parser.parse_args()
read_file = args.read
out_dir = args.out

max_templates = []
pre_dicts = []
with open(read_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        max_template = selected_max_certainly(data)
        max_templates.append(max_template)
        pre_dict["pred_certainly_json"] = max_template
        pre_dicts.append(pre_dict)


print("Done")
with open(out_dir, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')