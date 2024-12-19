import sys
import json
from BETTER_Granular_Class_simplified import *

def most_relevant_string(span_list):
    preprocess_span_list = []
    for span in span_list:
        if "synclass" in span and "string" in span:
            preprocess_span_list.append({"string":span["string"], "synclass":span["synclass"]})
        elif "string" in span:
            preprocess_span_list.append({"string":span["string"], "synclass":None})
    preprocess_span_list.sort(key=lambda x: len(x["string"]), reverse=True)
    for span in preprocess_span_list:
        if "synclass" in span:
            if span["synclass"] == "name":
                return span
    for span in preprocess_span_list:
        if "synclass" in span:
            if span["synclass"] == "nominal":
                return span
    for span in preprocess_span_list:
        if "synclass" in span:
            if span["synclass"] == "pronoun":
                return span
    return preprocess_span_list[0]



def apply_template(entry):
    #Horror absoluto codigo hau, sentitzet
    templates = entry["annotation-sets"]["basic-events"]["granular-templates"]
    pre_template_list = []
    for template in templates:
        pre_template = eval(templates[template]["template-type"])
        att_dict = {}
        #for template_key in pre_template.__fields__.keys():
        #    if template_key != "template_type":
        #        att_dict[template_key] = None
        for temp_key in templates[template]:
            pre_key = temp_key.replace("-", "_")
            data = templates[template][temp_key]
            if isinstance(data, list):
                pre_data = []
                for instances in data:
                    pre_ins = None
                    for key_ins in instances:
                        if key_ins == "irrealis":
                            continue
                        elif key_ins == "time-attachments":
                            continue
                        elif key_ins == "ssid":
                            str_ssid = instances[key_ins]
                            corref = entry["annotation-sets"]["basic-events"]["span-sets"][str_ssid]["spans"]
                            selected_span = most_relevant_string(corref)
                            pre_ins =span.model_validate(selected_span)
                            print(temp_key)
                            print(selected_span)
                        elif key_ins == "event-id":
                            str_event = instances[key_ins]
                            event_data = entry["annotation-sets"]["basic-events"]["events"][str_event]
                            event_span_data = entry["annotation-sets"]["basic-events"]["span-sets"][event_data["anchors"]]["spans"]
                            selected_anchor = most_relevant_string(event_span_data)
                            pre_anchor = span.model_validate(selected_anchor)
                            pre_ins = event.model_validate({"anchors":pre_anchor, "event_type":event_data["event-type"]})
                        else:
                            print("ERROR! Key not found: ", key_ins)
                            exit(-1)
                    if pre_ins is not None:
                        pre_data.append(pre_ins)
                att_dict[pre_key] = pre_data

            elif pre_key == "template_id" or pre_key == "template_type":
                continue

            elif pre_key == "template_anchor":
                anchor_data = data
                anchor_span_data = entry["annotation-sets"]["basic-events"]["span-sets"][anchor_data]["spans"]
                selected_anchor = most_relevant_string(anchor_span_data)
                pre_anchor = span.model_validate(selected_anchor)
                att_dict[pre_key] = pre_anchor
            elif isinstance(data, bool) or isinstance(data, str):
                att_dict[pre_key] = data

            else:
                print("ERROR! Temp_key not found:", pre_key)
                exit(-1)
        for key in att_dict:
            if key not in pre_template.model_fields.keys():
                print("ERROR! Key not found in template:", key)
                assert False
        pre_template = pre_template.model_validate(att_dict)
        pre_template_list.append(pre_template)
    preprocess_template = Template.model_validate({"templates":pre_template_list},strict=True)
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
    output_files = ['phase2/phase2.granular.eng.preprocess-dev-simplified.jsonl', 'phase2/phase2.granular.eng.preprocess-test-simplified.jsonl', 'phase2/phase2.granular.eng.preprocess-train-simplified.jsonl']
    for i in range(len(input_files)):
        input_file = input_files[i]
        output_file = output_files[i]
        preprocess_phase2_english(input_file, output_file)