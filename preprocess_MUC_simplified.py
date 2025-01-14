import json
import os


file_paths = []
output_file_paths = []
languages = ["en","ar", "fa", "ko", "ru", "zh"]
for language in languages:
    path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/train.jsonl"
    path_write = "multimuc/data/multimuc_v1.0/corrected/" + language + "/train_simplified_preprocess.jsonl"
    if os.path.exists(path_write):
        os.remove(path_write)
    file_paths.append(path_read)
    output_file_paths.append(path_write)



for i in range(len(file_paths)):
    with open(file_paths[i], 'r') as file:
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
                            max_length = ""
                            for correference in slot:
                                name = correference[0]
                                if len(name) > len(max_length):
                                    max_length = name
                            if new_slot != "":
                                new_slot.append(max_length)
                        new_template[key] = new_slot
                new_templates.append(new_template)
            write_data = {"docid": data["docid"],"doctext": data["doctext"], "templates": new_templates}
            with open(output_file_paths[i], 'a') as output_file:
                output_file.write(json.dumps(write_data, ensure_ascii=False) + '\n')
    print(max)
for file in output_file_paths:
    with open(file, 'r') as f:
        for line in f:
            st = json.loads(line)

print("funtzionatzeu")
                