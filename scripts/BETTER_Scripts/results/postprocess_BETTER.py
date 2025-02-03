import json
from types import NoneType
import sys
import os
sys.path.append("class_data")
from BETTER_Granular_Class import *
import BETTER_Granular_Class_simplified as simp_class
import pylcs
sys.path.append("scripts/BETTER_Scripts/preprocess")
from simplified2normal import simplified2normal

def detect_largest_coincidency(document, span_string):
    res = pylcs.lcs_string_idx(span_string, document)
    result = ''.join([document[i] for i in res if i != -1])
    start = 0
    end = 0
    if result == "":
        print("ERROR! No match found")
    else:
        lag_res = [i for i in res if i != -1]
        start = lag_res[0]
        end = lag_res[-1] + 1
    return result, start, end



def event_ids(event,prev_event_ids,last_key="event-0"):
    for key in prev_event_ids:
        saved_event = prev_event_ids[key]
        if event["anchors"] == saved_event["anchors"] and event["agents"] == saved_event["agents"] and event["patients"] == saved_event["patients"] and event["event-type"] == saved_event["event-type"]:
            return key
    
    last_key_int = int(last_key.split("-")[1])
    last_key_int = last_key_int + 1
    str_last_key = "event-"+str(last_key_int)
    return str_last_key

def spans_ids(spans,prev_spans_ids,document,last_key="ss-0",is_anchor=False):
    for span in spans["spans"]:
        if "synclass" in span:
            if span["synclass"] == None:
                span["synclass"] = "pronoun"
            elif span["synclass"] not in ['name', 'nominal', 'pronoun', 'event-anchor', 'template-anchor', 'time-mention', 'duration-mention']:
                span["synclass"] = "pronoun"
        span["string"], span["start"], span["end"] = detect_largest_coincidency(document,span["string"])
        if is_anchor:
            span["synclass"] = "event-anchor"
            span["anchor-string"] = True
    for key in prev_spans_ids:
        saved_span = prev_spans_ids[key]["spans"]
        if spans["spans"] == saved_span:
            return key
    last_key_int = int(last_key.split("-")[1])
    last_key_int = last_key_int + 1
    str_last_key = "ss-"+str(last_key_int)
    return str_last_key

def create_all_ids(data,document,simplified=False):
    all_spans = {}
    all_events = {}
    new_templates = {}
    template_counter = 0
    last_span_key = "ss-0"
    last_event_key = "event-0"
    compare_ids = lambda x,y: int(x.split("-")[1]) > int(y.split("-")[1])

    if simplified:
        data = simplified2normal(data)

    for template in data:
        new_template = {}
        for temp_key in template:
            new_temp_key = temp_key.replace("_","-")
            if isinstance(template[temp_key],list):
                temp_list = template[temp_key]
                new_template[new_temp_key] = []
                for dict_list in temp_list:
                    current_slot = {}
                    for key_dict in dict_list:
                        if key_dict == "span":
                            new_key_span = spans_ids(dict_list[key_dict],all_spans,document,last_span_key)
                            if compare_ids(new_key_span,last_span_key):
                                last_span_key = new_key_span
                                all_spans[new_key_span] = {"spans":dict_list[key_dict]["spans"],"ssid":new_key_span}
                            current_slot["ssid"] = new_key_span
                        elif key_dict == "event":
                            event = dict_list[key_dict]
                            event_dict = {"anchors":None,"agents":[],"patients":[],"event-type":event["event_type"],"ref-events":[]}
                            if "SoA" in event_dict["event-type"]:
                                event_dict["state-of-affairs"] = True
                            else:
                                event_dict["state-of-affairs"] = False
                            new_key_span = spans_ids(event["anchors"],all_spans,document,last_span_key,is_anchor=True)
                            if compare_ids(new_key_span,last_span_key):
                                last_span_key = new_key_span
                                all_spans[new_key_span] = {"spans":event["anchors"]["spans"],"ssid":new_key_span}
                            event_dict["anchors"]=new_key_span
                            for event_key in ["agents","patients"]:
                                for spans in event[event_key]:
                                    new_key_span = spans_ids(spans,all_spans,document,last_span_key)
                                    if compare_ids(new_key_span,last_span_key):
                                        last_span_key = new_key_span
                                        all_spans[new_key_span] = {"spans":spans["spans"],"ssid":new_key_span}
                                    event_dict[event_key].append(new_key_span)
                            new_key_event = event_ids(event_dict,all_events,last_event_key)
                            event_dict["eventid"] = new_key_event
                            if compare_ids(new_key_event,last_event_key):
                                last_event_key = new_key_event
                                all_events[new_key_event] = event_dict
                            current_slot["event-id"] = new_key_event
                            
                        elif key_dict == "irrealis":
                            if not isinstance(dict_list[key_dict],NoneType):
                                current_slot["irrealis"] = dict_list[key_dict]
                        elif key_dict == "time_attachments":
                            if not isinstance(dict_list[key_dict],NoneType):
                                current_slot["time-attachments"] = []
                                for spans in dict_list[key_dict]:
                                    new_key_span = spans_ids(spans,all_spans,document,last_span_key)
                                    if compare_ids(new_key_span,last_span_key):
                                        last_span_key = new_key_span
                                        all_spans[new_key_span] = {"spans":spans["spans"],"ssid":new_key_span}
                                    current_slot["time-attachments"].append(new_key_span)
                        else:
                            print("ERROR! Key not found: ", key_dict)
                    new_template[new_temp_key].append(current_slot)
            elif temp_key == "template_anchor":
                new_key_span = spans_ids(template[temp_key],all_spans,document,last_span_key)
                if compare_ids(new_key_span,last_span_key):
                    last_span_key = new_key_span
                    all_spans[new_key_span] = {"spans":template[temp_key]["spans"],"ssid":new_key_span}
                new_template[new_temp_key] = new_key_span
            elif not isinstance(template[temp_key],NoneType):
                new_template[new_temp_key] = template[temp_key]

        template_counter += 1
        id = "template-"+str(template_counter)
        new_template["template-id"] = id
        new_templates[id] = new_template

    result = {"events":all_events,"granular-templates":new_templates,"span-sets":all_spans}
    return result

def postprocess(file_path, simplified=False):
    postprocess_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            if simplified:
                template_data = simp_class.Template.model_validate({"templates":data["templates"]}).model_dump()
            else:
                template_data = Template.model_validate({"templates":data["templates"]}).model_dump()
            postprocess_template = create_all_ids(template_data["templates"],data["doctext"],simplified)
            postprocess_dict[data["docid"]] = {"annotation-sets":{"basic-events":postprocess_template},"doc-id":data["docid"],"entry-id":data["docid"],"segment-text":data["doctext"]}
    header = {"corpus-id":"Phase 2 Granular English 16 Dec 2021, Provided Devtest Ref (Obfuscated)", "entries":postprocess_dict, "format-type":"bp-corpus", "format-version":"v10"}
    return header

def write_postprocessed_data(input_file_path, output_file_path, simplified=False):
    
    postprocessed_data = postprocess(input_file_path, simplified)
    output_folder = os.path.dirname(output_file_path)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_file_path, 'w') as file:
        json.dump(postprocessed_data, file, indent=4)


if __name__ == "__main__":
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    write_postprocessed_data(input_file_path, output_file_path)
    