import json
import os

file_path = 'multimuc/data/multimuc_v1.0/en/train.jsonl'
output_file_path = 'multimuc/data/multimuc_v1.0/en/train_preprocess.jsonl'
if os.path.exists(output_file_path):
    os.remove(output_file_path)

with open(file_path, 'r') as file:
    max = 0
    for line in file:
        data = json.loads(line)
        new_templates = []
        for template in data["templates"]:
            if max < len(template):
                max = len(template)
            new_template = {"incident_type": template["incident_type"]}
            for key in template.keys():
                if key != "incident_type":
                    new_slot = []
                    for slot in template[key]:
                        all_correferences = []
                        for correference in slot:
                            name = correference[0]
                            all_correferences.append(name)
                        new_slot.append(all_correferences)
                    new_template[key] = new_slot
            new_templates.append(new_template)
        write_data = {"docid": data["docid"],"doctext": data["doctext"], "templates": new_templates}
        with open(output_file_path, 'a') as output_file:
            output_file.write(json.dumps(write_data) + '\n')
print(max)
                