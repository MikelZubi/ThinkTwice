import json
import os
from MUC_Class import *


file_paths = []
output_file_paths = []
languages = ["en","ar", "fa", "ko", "ru", "zh"]
for language in languages:
    path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/train.jsonl"
    path_write = "multimuc/data/multimuc_v1.0/corrected/" + language + "/train_preprocess.jsonl"
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
            incident_types = []
            entity_list = []
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
                        entity_list.append(all_correferences)
                    else:
                        incident_types.append(template["incident_type"])
                new_templates.append(Template.model_validate(new_template))
            pre_incident_types = Incident_Types.model_validate({"incident_types":incident_types})
            pre_entity_list = Entities.model_validate({"entities":entity_list})
            pre_template = Base.model_validate({"templates":new_templates})
            write_data = {"docid": data["docid"],"doctext": data["doctext"], "templates": pre_template.model_dump(), "entities": pre_entity_list.model_dump(), "incident_types": pre_incident_types.model_dump()}
            with open(output_file_paths[i], 'a') as output_file:
                output_file.write(json.dumps(write_data, ensure_ascii=False) + '\n')
    print(max)
                