import json
import os
import argparse
from collections import defaultdict

import sys
import json
import copy as cp
from typing import OrderedDict, List, Union, Tuple, Optional, Callable, Dict

sys.path.append('multimuc/iterx/src')

from iterx.metrics.muc.ceaf_rme import generate_scoring_structures, IterXTemplate, SCORER_CONSTRUCTOR
from iterx.metrics.muc.ceaf_rme import ScoreFunction

from pathlib import Path
from typing import Annotated
import typer
from iterx.metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric, \
    load_references, print_prediction_comparison
from iterx.metrics.muc.ceaf_rme import ScoreFunction
import os
import argparse
import numpy as np
import copy as cp 



def score(
        pred_data: Dict,
        ref_data: List,
        dataset: Annotated[DatasetKind, typer.Option()] = DatasetKind.MUC,
        ignore_no_template_doc: Annotated[bool, typer.Option(
            help='Whether to ignore documents without any templates during scoring.'
        )] = False,
        sanitize_special_chars: Annotated[bool, typer.Option(
            help='Whether to sanitize special characters in the predictions and references.'
        )] = True,
        scirex_merge_mentions: Annotated[bool, typer.Option(
            help='Whether to merge mentions in the SciREX predictions to form entities.'
        )] = True,
        scorer: Annotated[ScoreFunction, typer.Option(
            help='The scoring function to use.'
        )] = ScoreFunction.Phi3,
        file_type: Annotated[PredictionFileType, typer.Option(
            help='The type of the prediction file.'
        )] = PredictionFileType.GTT,
        remove_span_whitespace: Annotated[bool, typer.Option(
			help='Whether to concatenate all the spans in a mention to form a single span.'
        )] = False,
):
    """Score a prediction file against a reference file."""
    normalize_role = True if dataset == DatasetKind.MUC else False
    convert_doc_id = True if file_type == PredictionFileType.GTT and dataset == DatasetKind.MUC else False
    # SciREX postprocessing should have been moved out of the scorer.
    #Honek itten duna ya iñe
    predictions = load_predictions(pred_file=pred_data,
                                   dataset=dataset,
                                   file_type=file_type,
                                   normalize_role=normalize_role,
                                   remove_span_whitespace=remove_span_whitespace,
                                   scirex_merge_mentions=scirex_merge_mentions,
                                   compute_metrics=True)
    #print(predictions)
    #print("\n\nSALTO\n\n")
    metric = load_metric(dataset_kind=dataset,
                         doc_path={"dev": ref_data},
                         ignore_no_template_doc=ignore_no_template_doc,
                         sanitize_special_chars=sanitize_special_chars,
                         scorer_type=scorer,
                         convert_doc_id=convert_doc_id,
                         compute_metrics=True)
    #print(metric.references)
    metric(
        predictions=predictions,
        pred_src_file="dev",
        dedup=False,
        cluster_substr=False,
        normalize_role=normalize_role
    )
    results = metric.get_metric(reset=True)
    return results 

def simplify_template(templates):
    new_templates = []
    for template in templates:
        new_template = {}
        for key in template.keys():
            if key=="incident_type":
                new_template[key] = template[key]
            elif template[key]==[]:
                new_template[key] = template[key]
            else:
                new_template[key] = []
                for element in template[key]:
                    new_template[key].append(element[0])
        new_templates.append(new_template)
    dict_templates = {"templates": new_templates}
    return json.dumps(dict_templates, ensure_ascii=False)

def remove_only_incident_temp(templates):
    new_templates = []
    for template in templates:
        if not (template["PerpInd"] == [] and
                template["PerpOrg"] == [] and
                template["Target"] == [] and
                template["Victim"] == [] and
                template["Weapon"] == []):
            new_templates.append(template)
    return new_templates

def postprocess(id,pred,label,document):
    lower_doc = document.lower()
    post_preds = {}
    post_labels = []
    pred_json = {"pred_templates": pred, "gold_templates": label}
    pred_id = str(
                int(id.split("-")[0][-1]) * 10000
                + int(id.split("-")[-1]))
    post_preds[pred_id] = pred_json
    indx_labels = []
    for template in label:
        post_processed = {}
        for key in template.keys():
            if key != "incident_type" and template[key] != []:
                post_processed[key]=[[[elem[0].lower(),lower_doc.find(elem[0].lower())]] if elem[0].lower() in lower_doc else [] for elem in template[key]]
            else:
                post_processed[key]=template[key]
        indx_labels.append(post_processed)
    post_labels.append({"docid": id, "templates": indx_labels})
    return post_preds, post_labels    

def template_voting(templates,document):
    templates = list(filter((["ERROR"]).__ne__, templates))
    templates = list(filter(([["ERROR"]]).__ne__, templates))
    cleaned_templates = [remove_only_incident_temp(template) for template in templates]
    
    empty_counter = 0
    for template in cleaned_templates:
        if len(template) == 0:
            empty_counter += 1
    if empty_counter * 2 > len(templates):
        return []
    templates = list(filter(([]).__ne__, templates))
    mean_templates = []
    for template_a in cleaned_templates:
        current_f1s = 0.0
        for template_b in cleaned_templates:
            post_b, post_a = postprocess("TST1-MUC3-0001", template_b, template_a, document)
            score_result = score(pred_data=post_b, ref_data=post_a)
            current_f1 = score_result["iterx_muc_slot_f1"]
            current_f1s += current_f1
        mean_templates.append(current_f1s / len(cleaned_templates))
    print(mean_templates)
    maximum_num = np.argmax(mean_templates)
    template = cleaned_templates[maximum_num]
    print(template)
    return template

    
#Argument parser
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--read', dest='read', type=str)
parser.add_argument("--n", dest='n', type=int)
parser.set_defaults(n=1)

args = parser.parse_args()
read_file = args.read

outputs = []
break_counter = 0
break_count = 0
with open(read_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        pre_dict = data
        template_list = cp.deepcopy(data["pred_json"])
        max_template = template_voting(data["pred_json"],data["doctext"])
        post_template = simplify_template(max_template)
        for n in range(args.n):
            if max_template not in template_list:
                break_counter += args.n - n - 1 
                break_count += 1
                break
            template_idx = template_list.index(max_template)
            reasoning = data["pred_reasoning"][template_idx]
            template_list.pop(template_idx)
            outputs.append({"docid": data["docid"], "completion": "<think>\n" + reasoning + "</THINK_TOKENA>" + post_template, "doctext": data["doctext"]})

print("BREAK COUNTER OSOA " + str(break_counter))
print("BREAK COUNTER KOP " + str(break_count))
out_path = "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/train_best"+str(args.n)+".jsonl"
with open(out_path, 'w') as file:
    for line in outputs:
        file.write(json.dumps(line, ensure_ascii=False) + "\n")




