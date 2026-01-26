import json
import os
import argparse
import argparse
import numpy as np


def postprocess(id, pred, label, document):
    lower_doc = document.lower()
    post_preds = {}
    post_labels = []
    pred_json = {"pred_templates": pred, "gold_templates": label}
    pred_id = str(
        int(id.split("-")[0][-1]) * 10000
        + int(id.split("-")[-1]))
    post_preds[pred_id] = pred_json
    indx_labels = []
    for template in label:
        post_processed = {}
        for key in template.keys():
            if key != "incident_type" and template[key] != []:
                post_processed[key] = [
                    [[elem[0].lower(), lower_doc.find(elem[0].lower())]] if elem[0].lower() in lower_doc else [] for
                    elem in template[key]]
            else:
                post_processed[key] = template[key]
        indx_labels.append(post_processed)
    post_labels.append({"docid": id, "templates": indx_labels})
    return post_preds, post_labels


def remove_errors(all_templates):
    return [template for template in all_templates if "ERROR" not in template]


def template_voting(templates, document):
    cleaned_templates = remove_errors(templates)
    if cleaned_templates == []:
        return {"templates": []}
    # templates = list(filter(([]).__ne__, templates))
    mean_templates = []
    for template_a in cleaned_templates:
        current_scores = 0.0
        for template_b in cleaned_templates:
            if template_a == template_b:
                current_scores += 1.0
        mean_templates.append(current_scores / len(cleaned_templates))
    print(mean_templates)
    maximum_num = np.argmax(mean_templates)
    template = cleaned_templates[maximum_num]
    print(template)
    return {"templates": template}


# Argument parser
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
        pre_dict = {}
        pre_dict["docid"] = data["docid"]
        pre_dict["doctext"] = data["doctext"]
        pre_dict["templates"] = data["templates"]
        print(f"Processing document: {data['docid']}")
        max_template = template_voting(data["pred_json"], data["doctext"])
        max_templates.append(max_template)
        pre_dicts.append(pre_dict)

correct_in_templates = 0
for idx, outputs in enumerate(max_templates):
    post_templates = []
    lower_doc = pre_dicts[idx]["doctext"].lower()
    for template in outputs["templates"]:
        post_processed = {}
        if isinstance(template, list):
            post_processed = template
            continue
        for key in template.keys():
            if key != "incident_type" and template[key] != []:
                for elem in template[key]:
                    if elem[0].lower() not in lower_doc:
                        correct_in_templates += 1
                post_processed[key] = [[elem[0].lower()] for elem in template[key] if elem[0].lower() in lower_doc]
            else:
                post_processed[key] = template[key]
        post_templates.append(post_processed)
    pre_dicts[idx]["pred_json_scorer"] = post_templates

print("Done")
print(f"Correct in templates: {correct_in_templates}")
with open(out_dir, 'w') as output_file:
    for line in pre_dicts:
        output_file.write(json.dumps(line, ensure_ascii=False) + '\n')



