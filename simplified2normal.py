from BETTER_Granular_Class import *
import copy as cp 

def simplified2normal(simp_template):
    new_template = cp.deepcopy(simp_template)
    for i in range(len(simp_template)):
        for key in simp_template[i]:
            if key == "template_anchor":
                new_spans = spans.model_validate({"spans":[simp_template[i][key]]}).model_dump()
                new_template[i][key] = new_spans
            elif isinstance(simp_template[i][key], list):
                for j in range(len(simp_template[i][key])):
                    if "string" in simp_template[i][key][j]: #Span
                        new_spans = spans.model_validate({"spans":[simp_template[i][key][j]]}).model_dump()
                        new_template[i][key][j] = span_set.model_validate({"span": new_spans}).model_dump()
                    else: #Event
                        new_event = {"anchors": spans.model_validate({"spans":[simp_template[i][key][j]["anchors"]]}).model_dump(),
                                    "agents": [],
                                    "patients": [],
                                    "event_type": simp_template[i][key][j]["event_type"]}
                        new_template[i][key][j] = event_set.model_validate({"event":new_event}).model_dump()
           
    new_template = Template.model_validate({"templates":new_template})
    result = new_template.model_dump()["templates"]
    return result
            