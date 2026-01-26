import BETTER_Granular_Class_string as BETTER_string
import BETTER_Granular_Class_simplified as BETTER_simpl
import argparse
import copy as cp

EVENT_SYNCLASS = {"Assistance_needed": "Aid-Needs",
 "Assistance_provided": "Provide-Aid",
 "outcome": 'Close-Schools',
 "settlement_status_event_or_SoA": 'Employ-Workers',
 "event_or_SoA_at_origin": 'War-Event-or-SoA',
 "human_displacement_event": 'Refugee-Movement',
 "human_displacement_events": 'Refugee-Movement',
 "repair": 'Construct-Project',
 "disease_outbreak_events": "Disease-Outbreak",
 "declare_emergency": "Declare-Emergency",
 "announce_disaster_warnings": "Declare-Emergency",
 "related_natural_phenomena": 'Natural-Phenomenon-Event-or-SoA',
 "assistance_needed": "Aid-Needs",
 "assistance_provided": "Provide-Aid",
 "rescue_events": "Rescue",
 }

def str2simp(data):
    new_data = cp.deepcopy(data)
    for i in range(len(data)):
        anchor = data[i]["template_anchor"]
        for key in data[i]:
            if key == "template_anchor":
                new_span = BETTER_simpl.span.model_validate({"string": data[i][key], "synclass": 'template-anchor'}).model_dump()
                new_data[i][key] = new_span
            elif isinstance(data[i][key], list):
                for j in range(len(data[i][key])):
                    #All synclass = None except the anchor
                    if anchor == data[i][key][j]:
                        new_span = BETTER_simpl.span.model_validate({"string":data[i][key][j],"synclass": 'template-anchor'}).model_dump()
                    else:
                        new_span = BETTER_simpl.span.model_validate({"string":data[i][key][j], "synclass": None}).model_dump()
                    if key in EVENT_SYNCLASS:
                        new_event = {"anchors": new_span,
                             "event_type": EVENT_SYNCLASS[key]} 
                        new_data[i][key][j] = BETTER_simpl.event.model_validate(new_event).model_dump()
                    else:
                        new_data[i][key][j] = new_span

    new_data = BETTER_simpl.Template.model_validate({"templates":new_data})
    result = new_data.model_dump()["templates"]
    return result