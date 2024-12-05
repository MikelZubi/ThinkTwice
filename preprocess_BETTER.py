import sys
import json
from BETTER_Granular_Class import *


def apply_template(entry):
    #Horror absoluto codigo hau, sentitzet
    templates = entry["annotation-sets"]["basic-events"]["granular-templates"]
    pre_template_list = []
    for template in templates:
        pre_template = eval(templates[template]["template-type"])
        att_dict = {}
        for template_key in pre_template.__fields__.keys():
            if template_key != "template_type":
                att_dict[template_key] = None
        for temp_key in templates[template]:
            pre_key = temp_key.replace("-", "_")
            data = templates[template][temp_key]
            if isinstance(data, list):
                pre_data = []
                for instances in data:
                    pre_ins = {"irrealis": None, "time_attachments": None}
                    is_span_list = True #True if spans, False if events
                    for key_ins in instances:
                        if key_ins == "irrealis":
                            pre_ins["irrealis"] = instances[key_ins]
                        elif key_ins == "time-attachments":
                            time_attachment_list_span = instances[key_ins]
                            pre_time_attachment_list = []
                            for time_attachment_span in time_attachment_list_span:
                                time_attachment_span_data = entry["annotation-sets"]["basic-events"]["span-sets"][time_attachment_span]["spans"]
                                pre_corref = []
                                for time_attachment_span_dict in time_attachment_span_data:
                                    if "synclass" not in time_attachment_span_dict:
                                        time_attachment_span_dict["synclass"] = None
                                    time_attachment_span = span.model_validate({"string":time_attachment_span_dict["string"], "synclass":time_attachment_span_dict["synclass"]})
                                    pre_corref.append(time_attachment_span)
                                pre_time_attachment_list.append(spans.model_validate({"spans":pre_corref}))
                            pre_ins["time_attachments"] = pre_time_attachment_list
                        elif key_ins == "ssid":
                            str_ssid = instances[key_ins]
                            corref = entry["annotation-sets"]["basic-events"]["span-sets"][str_ssid]["spans"]
                            pre_corref = [] 
                            for corref_dict in corref:
                                if "synclass" not in corref_dict:
                                    corref_dict["synclass"] = None
                                correference = span.model_validate({"string":corref_dict["string"], "synclass":corref_dict["synclass"]})
                                pre_corref.append(correference)
                            pre_ins["span"] =spans.model_validate({"spans":pre_corref})
                            is_span_list = True
                        elif key_ins == "event-id":
                            str_event = instances[key_ins]
                            event_data = entry["annotation-sets"]["basic-events"]["events"][str_event]
                            event_span_data = entry["annotation-sets"]["basic-events"]["span-sets"][event_data["anchors"]]["spans"]
                            pre_corref = []
                            for event_span_dict in event_span_data:
                                if "synclass" not in event_span_dict:
                                    event_span_dict["synclass"] = None
                                event_span = span.model_validate({"string":event_span_dict["string"], "synclass":event_span_dict["synclass"]})
                                pre_corref.append(event_span)
                            pre_anchor = spans.model_validate({"spans":pre_corref})
                            event_agent_data = event_data["agents"]
                            pre_event_agent_data = []
                            for event_agent_dict in event_agent_data:
                                event_agent_span_data = entry["annotation-sets"]["basic-events"]["span-sets"][event_agent_dict]["spans"]
                                pre_corref = []
                                for event_agent_span_dict in event_agent_span_data:
                                    if "synclass" not in event_agent_span_dict:
                                        event_agent_span_dict["synclass"] = None
                                    event_agent_span = span.model_validate({"string":event_agent_span_dict["string"], "synclass":event_agent_span_dict["synclass"]})
                                    pre_corref.append(event_agent_span)
                                pre_event_agent_data.append(spans.model_validate({"spans":pre_corref}))
                            event_patient_data = event_data["patients"]
                            pre_event_patient_data = []
                            for event_patient_dict in event_patient_data:
                                event_patient_span_data = entry["annotation-sets"]["basic-events"]["span-sets"][event_patient_dict]["spans"]
                                pre_corref = []
                                for event_patient_span_dict in event_patient_span_data:
                                    event_patient_span = span.model_validate({"string":event_patient_span_dict["string"], "synclass":event_patient_span_dict["synclass"]})
                                    pre_corref.append(event_patient_span)
                                pre_event_patient_data.append(spans.model_validate({"spans":pre_corref}))
                            
                            pre_ins["event"] = event.model_validate({"anchors":pre_anchor, "agents":pre_event_agent_data, "patients":pre_event_patient_data, "event_type":event_data["event-type"]})
                            is_span_list = False
                        else:
                            print("ERROR! Key not found: ", key_ins)
                            exit(-1)
                    if is_span_list:
                        pre_data.append(span_set.model_validate(pre_ins))
                    else:
                        pre_data.append(event_set.model_validate(pre_ins))
                att_dict[pre_key] = pre_data

            elif pre_key == "template_id" or pre_key == "template_type":
                continue

            elif pre_key == "template_anchor":
                anchor_data = data
                anchor_span_data = entry["annotation-sets"]["basic-events"]["span-sets"][anchor_data]["spans"]
                pre_corref = []
                for anchor_span_dict in anchor_span_data:
                    anchor_span = span.model_validate({"string":anchor_span_dict["string"], "synclass":anchor_span_dict["synclass"]})
                    pre_corref.append(anchor_span)
                pre_anchor = spans.model_validate({"spans":pre_corref})
                att_dict[pre_key] = pre_anchor
            elif isinstance(data, bool) or isinstance(data, str):
                att_dict[pre_key] = data

            else:
                print("ERROR! Temp_key not found:", pre_key)
                exit(-1)
        pre_template = pre_template.model_validate(att_dict)
        pre_template_list.append(pre_template)
    preprocess_template = Template.model_validate({"templates":pre_template_list})
    return preprocess_template


def preprocess_phase2_english(input_file, output_file):
    # Initialize the BETTER template

    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Process each entry in the data
    processed_data = []
    for entry_key in data["entries"]:
        entry = data["entries"][entry_key]
        processed_entry = apply_template(entry)
        processed_data.append({"templates":processed_entry.model_dump()["templates"],"doctext":entry["segment-text"],"docid":entry_key})

    # Write the processed data to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in processed_data:
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write('\n')
if __name__ == "__main__":
    input_files = ['phase2/phase2.granular.eng.dev.json', 'phase2/phase2.granular.eng.test.json', 'phase2/phase2.granular.eng.train.json']
    output_files = ['phase2/phase2.granular.eng.preprocess-dev.jsonl', 'phase2/phase2.granular.eng.preprocess-test.jsonl', 'phase2/phase2.granular.eng.preprocess-train.jsonl']
    for i in range(len(input_files)):
        input_file = input_files[i]
        output_file = output_files[i]
        preprocess_phase2_english(input_file, output_file)