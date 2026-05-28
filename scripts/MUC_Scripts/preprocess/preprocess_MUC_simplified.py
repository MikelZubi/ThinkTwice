import json
import os
import sys
from tqdm import tqdm
sys.path.append("class_data")
from MUC_Class_simplified import Base

def delete_correferences_simplified(templates):
    new_templates = []
    for template in templates:
        new_template = {}
        for key in template.keys():
            if key=="incident_type":
                new_template[key] = template[key]
                if "attack" in template[key] and template[key]!="attack":
                    new_template[key] = "attack"
            elif template[key]==[]:
                new_template[key] = template[key]
            else:
                new_template[key] = []
                for element in template[key]:
                    new_template[key].append(element[0][0])
        new_templates.append(new_template)
    Base.model_validate({"templates": new_templates})
    return new_templates


file_paths = []
output_file_paths = []
languages = ["en", "ar", "fa", "ko", "ru", "zh"]
map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}
for split in ["test", "train", "dev"]:
    for language in languages:
        path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/"+split+".jsonl"
        path_write = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
        if os.path.exists(path_write):
            os.remove(path_write)
        file_paths.append(path_read)
        output_file_paths.append(path_write)



for i in range(len(file_paths)):
    with open(file_paths[i], 'r') as file:
        new_lines = []
        for line in file:
            data = json.loads(line)
            new_templates = []
            new_templates_raw = []
            incident_types = []
            simplfied_templates = delete_correferences_simplified(data["templates"])
            write_data = {"docid": data["docid"], "doctext": " ".join(data["doctext"].split()), "simp_templates": simplfied_templates, "templates": data["templates"]}
            new_lines.append(write_data)
        with open(output_file_paths[i], 'w') as output_file:
            for line in new_lines:
                output_file.write(json.dumps(line, ensure_ascii=False) + '\n')
          