import json
from tqdm import tqdm


#Simplified the template removing double lists and convert it to json
def simplify_template(templates):
    new_templates = []
    for template in templates:
        new_template = {}
        for key in template.keys():
            if key=="incident_type":
                new_template[key] = template[key]
            elif template[key]==[]:
                new_template[key] = template[key]
            else:
                new_template[key] = []
                for element in template[key]:
                    new_template[key].append(element[0])
        new_templates.append(new_template)
    return new_templates

def only_incident_template(templates):
    for template in templates:
        if not (template["PerpInd"] == [] and
                template["PerpOrg"] == [] and
                template["Target"] == [] and
                template["Victim"] == [] and
                template["Weapon"] == []):
            return False
    return True

def equal_templates(template1, template2):
    #print(template1)
    #print(template2)
    simp_t1 = simplify_template(template1)
    if len(simp_t1) != len(template2):
        #print("Different length")
        return False
        
    for t1, t2 in zip(simp_t1, template2):
        if t1["incident_type"] != t2["incident_type"]:
            #print("Different incident type")
            return False
        for key in t1.keys():
            if key != "incident_type":
                if len(t1[key]) != len(t2[key]):
                    #print(f"Different length for key '{key}'")
                    return False
                for elem_t1 in t1[key]:
                    found = False
                    for elems_t2 in t2[key]:
                        if elem_t1 in elems_t2[0]:
                            found = True
                            break
                    if not found:
                        #print(f"Element '{elem_t1}' not found in key '{key}'")
                        return False
    return True

def post_process_template(template):
    simp_template = simplify_template(template)
    new_template = {"templates": simp_template}
    return json.dumps(new_template, ensure_ascii=False)

import argparse

#Argument parser
parser = argparse.ArgumentParser(description='Arguments for the creation of the train dataset')
parser.add_argument("--read", dest="read", type=str)
parser.add_argument("--out", dest="out", type=str)
parser.set_defaults(out=None)
read = parser.parse_args().read
#READ GOLD
gold_path = "multimuc/data/multimuc_v1.0/corrected/en/train.jsonl"#IDATZI
ground_truths = []
ids = []
documents = []
labels = []
outputs = []
with open(gold_path, "r") as file:
    for line in file:
        data = json.loads(line)
        ground_truths.append(data["templates"])
        ids.append(data["docid"])
        documents.append(data["doctext"])
        label = {"docid": data["docid"], "templates": data["templates"]}
        labels.append(label)

completions = []
out_list = []
path = read
best_f1s = []
dis = 0
outputs = []
out_onlytemp = []
best_templates = {}
counter_correct = 0
counter_correct_empty = 0
counter_correct_incident_only = 0
gold_reasonings = []
gold_reasonings_ids = []
gold_reasonings_templates = []

with open(path, "r") as file:
    for line, gold, id, document in tqdm(zip(file,ground_truths,ids,documents), total=len(ground_truths)):
        data = json.loads(line)
        for template,reasoning in zip(data["pred_json"],data["pred_reasoning"]):
            if template != ["ERROR"] and template != [["ERROR"]]:
                #TODO: Beittu 

                if equal_templates(template,gold):
                    gold_reasonings.append(reasoning)
                    gold_reasonings_ids.append(id)
                    gold_reasonings_templates.append(template)
                    counter_correct += 1
                    if template == []:
                        counter_correct_empty += 1
                    elif only_incident_template(template):
                        counter_correct_incident_only += 1
                    else:
                        print(gold)
                        print(reasoning)
                    break

print("Counter Correct:", counter_correct)
print("Counter Correct Empty:", counter_correct_empty)
print("Counter Correct Incident Only:", counter_correct_incident_only)

out = parser.parse_args().out
if out:
    gold_save = "multimuc/data/multimuc_v1.0/corrected/en/train_simplified_preprocess.jsonl"
    out_data = []

    with open(gold_save, "r") as file:
        for line in file:
            data = json.loads(line)
            if data["docid"] in gold_reasonings_ids:
                idx = gold_reasonings_ids.index(data["docid"])
                post_template = post_process_template(gold_reasonings_templates[idx])
                reasoning = gold_reasonings[idx]
                out_data.append({"docid": data["docid"],
                                 "doctext": data["doctext"],
                                 "completion": "<think>\n" + reasoning + "</THINK_TOKENA>" + post_template})
            else:
                post_template = json.dumps(data["templates"], ensure_ascii=False)
                out_data.append({"docid": data["docid"],
                                 "doctext": data["doctext"],
                                 "completion": post_template})
    with open(out, "w") as file:
        for item in out_data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")