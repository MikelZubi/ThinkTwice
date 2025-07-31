import json
import os
import sys
sys.path.append("class_data")
from MUC_Class_simplified import *

file_paths = []
output_file_paths = []
languages = ["en","ar", "fa", "ko", "ru", "zh"]
map_field = {"PerpInd": "A person responsible for the incident. (PerpInd)", "PerpOrg": "An organization responsible for the incident. (PerpOrg)", "Target": "An inanimate object that was attacked. (Target)", "Victim": "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)", "Weapon": "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)"}
for split in ["test"]:
    for language in languages:
        path_read = "multimuc/data/multimuc_v1.0/corrected/"+language+"/"+split+".jsonl"
        path_write = "multimuc/data/multimuc_v1.0/corrected/" + language + "/"+split+"_simplified_preprocess.jsonl"
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
            new_templates_raw = []
            incident_types = []
            reasoning = "**Step 1: Identify the incident types:**\n\n"
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
                            if max_length != "":
                                new_slot.append(max_length)
                        new_template[key] = new_slot
                    else:
                        
                        if template[key] == 'bombing / attack' or template[key] == 'attack / bombing':
                            template[key] = 'attack'
                            new_template[key] = template[key]
                        incident_types.append(template["incident_type"])
                new_templates.append(Template.model_validate(new_template))
                new_templates_raw.append(new_template)
            continuation = True
            if len (incident_types) == 0:
                reasoning += "No incident type identified\n\n"
                continuation = False
            elif len (incident_types) == 1:
                reasoning += "Based on the information provided, the incident type is *" + incident_types[0] + "*\n\n"
            else:
                reasoning += "Based on the information provided, the incident types are "
                for incident in incident_types[:-1]:
                    reasoning += "*" + incident + "*, "
                reasoning += "and *" + incident_types[-1] + "*\n\n"
            reasoning += "**Step 2: Extract the entities of each incident types:**\n\n*"
            if not continuation:
                reasoning+= "Because there is not incident type identified, there are no entities to extract."

            for template_raw in new_templates_raw:
                reasoning += "For the incident type *" + template_raw["incident_type"] + "*, the entities are: "
                for key in template_raw.keys():
                    if key != "incident_type":
                        reasoning += "*" + map_field[key] + "*: "
                        if len(template_raw[key]) == 0:
                            reasoning += "No entity identified. "
                        elif len(template_raw[key]) == 1:
                            reasoning += "'" + template_raw[key][0] + "'. "
                        else:
                            for slot in template_raw[key][:-1]:
                                reasoning += "'" + slot + "', "
                            reasoning += "and '" + template_raw[key][-1] + "'. "
            pre_incident_types = Incident_Types.model_validate({"incident_types":incident_types})
            pre_template = Base.model_validate({"templates":new_templates})
            #write_data = {"docid": data["docid"],"doctext": " ".join(data["doctext"].split()), "templates": pre_template.model_dump(), "reasoning": reasoning, "incident_types": pre_incident_types.model_dump()}
            write_data = {"docid": data["docid"], "doctext": " ".join(data["doctext"].split()), "templates": pre_template.model_dump()}
            with open(output_file_paths[i], 'a') as output_file:
                output_file.write(json.dumps(write_data, ensure_ascii=False) + '\n')
    print(max)
for file in output_file_paths:
    with open(file, 'r') as f:
        for line in f:
            st = json.loads(line)                