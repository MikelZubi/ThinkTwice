import json
from types import NoneType



def event_ids(event,prev_event_ids,last_key="event-0"):
    for key in prev_event_ids:
        saved_event = prev_event_ids[key]
        if event == saved_event:
            return key
    
    last_key_int = int(last_key.split("-")[1])
    last_key_int = last_key_int + 1
    str_last_key = "event-"+str(last_key_int)
    prev_event_ids[str_last_key] = event
    return str_last_key

def spans_ids(spans,prev_spans_ids,last_key="ss-0"):
    for span in spans["spans"]:
        if span["synclass"] == None:
            del span["synclass"]
    for key in prev_spans_ids:
        saved_span = prev_spans_ids[key]["spans"]
        if spans == saved_span:
            return key
    
    last_key_int = int(last_key.split("-")[1])
    last_key_int = last_key_int + 1
    str_last_key = "ss-"+str(last_key_int)
    prev_spans_ids[str_last_key] = spans
    return str_last_key

def create_all_ids(data):
    all_spans = {}
    all_events = {}
    new_templates = {}
    template_counter = 0
    last_span_key = "ss-0"
    last_event_key = "event-0"
    compare_ids = lambda x,y: int(x.split("-")[1]) > int(y.split("-")[1])

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
                            new_key_span = spans_ids(dict_list[key_dict],all_spans,last_span_key)
                            if compare_ids(new_key_span,last_span_key):
                                last_span_key = new_key_span
                                all_spans[new_key_span] = {"spans":dict_list[key_dict]["spans"],"ssid":new_key_span}
                            current_slot["ssid"] = new_key_span
                        elif key_dict == "event":
                            event = dict_list[key_dict]
                            event_dict = {"anchors":None,"agents":[],"patients":[],"event-type":event["event_type"],"ref-events":[]}
                            new_key_span = spans_ids(event["anchors"],all_spans,last_span_key)
                            if compare_ids(new_key_span,last_span_key):
                                last_span_key = new_key_span
                                all_spans[new_key_span] = {"spans":event["anchors"]["spans"],"ssid":new_key_span}
                            event_dict["anchors"]=new_key_span
                            for event_key in ["agents","patients"]:
                                for spans in event[event_key]:
                                    new_key_span = spans_ids(spans,all_spans,last_span_key)
                                    if compare_ids(new_key_span,last_span_key):
                                        last_span_key = new_key_span
                                        all_spans[new_key_span] = {"spans":spans["spans"],"ssid":new_key_span}
                                    event_dict[event_key].append(new_key_span)
                            new_key_event = event_ids(event_dict,all_events)
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
                                for span in dict_list[key_dict]:
                                    new_key_span = spans_ids(span,all_spans,last_span_key)
                                    if compare_ids(new_key_span,last_span_key):
                                        last_span_key = new_key_span
                                        all_spans[new_key_span] = {"spans":spans["spans"],"ssid":new_key_span}
                                    current_slot["time-attachments"].append(new_key_span)
                        else:
                            print("ERROR! Key not found: ", key_dict)
                    new_template[new_temp_key].append(current_slot)
            elif temp_key == "template_anchor":
                new_key_span = spans_ids(template[temp_key],all_spans,last_span_key)
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

def postprocess(file_path):
    postprocess_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            postprocess_dict[data["docid"]] = {"annotation-sets":{"basic-events":create_all_ids(data["templates"])},"doc-id":data["docid"],"entry-id":data["docid"],"segment-text":data["doctext"]}
    header = {"corpus-id":"Phase 2 Granular English 16 Dec 2021, Provided Devtest Ref (Obfuscated)", "entries":postprocess_dict, "format-type":"bp-corpus", "format-version":"v10"}
    return header

if __name__ == "__main__":
    input_file_path = 'phase2/phase2.granular.eng.preprocess-dev.jsonl'
    output_file_path = 'phase2/phase2.granular.eng.postprocess-dev.json'
    
    postprocessed_data = postprocess(input_file_path)
    
    with open(output_file_path, 'w') as file:
        json.dump(postprocessed_data, file, indent=4)