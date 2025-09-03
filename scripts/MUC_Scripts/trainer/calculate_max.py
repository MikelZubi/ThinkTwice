from typing import List, Dict
import json
import sys
import numpy as np

sys.path.append('multimuc/iterx/src')

from iterx.metrics.muc.ceaf_rme import ScoreFunction

from typing import Annotated
import typer
from iterx.metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric
from iterx.metrics.muc.ceaf_rme import ScoreFunction

#RME Functions
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

#Simplified the template removing double lists and convert it to json
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
    simp_template = {"templates": new_templates}
    return json.dumps(simp_template, ensure_ascii=False)

def only_incident_template(templates):
    for template in templates:
        if not (template["PerpInd"] == [] and
                template["PerpOrg"] == [] and
                template["Target"] == [] and
                template["Victim"] == [] and
                template["Weapon"] == []):
            return False
    return True

def postprocess(id,pred,label):
    post_preds = {}
    post_labels = []
    pred_json = {"pred_templates": pred, "gold_templates": label}
    pred_id = str(
                int(id.split("-")[0][-1]) * 10000
                + int(id.split("-")[-1]))
    post_preds[pred_id] = pred_json
    post_labels.append({"docid": id, "templates": label})
    return post_preds, post_labels


import argparse

#Argument parser
parser = argparse.ArgumentParser(description='Arguments for the creation of the train dataset')
parser.add_argument("--read", dest="read", type=str)
parser.add_argument('--split', dest='split', type=str,
                    help='The split to use for the dataset.')
parser.set_defaults(dpo=False)
parser.set_defaults(iter=1)
parser.set_defaults(split="dev")
read = parser.parse_args().read
split = parser.parse_args().split
#READ GOLD
gold_path = "multimuc/data/multimuc_v1.0/corrected/en/"+split+".jsonl"#IDATZI
ground_truths = []
ids = []
documents = []
labels = []
outputs = []
with open(gold_path, "r") as file:
    for line in file:
        data = json.loads(line)
        ground_truths.append(data["templates"])
        new_id = "".join([data["docid"].split("-")[1], "-"+data["docid"].split("-")[0], "-"+data["docid"].split("-")[2]])
        ids.append(new_id)
        documents.append(data["doctext"])
        label = {"docid": new_id, "templates": data["templates"]}
        labels.append(label)

completions = []
out_list = []
path = read 
best_f1s = []
dis = 0
max = 0
outputs = []
out_onlytemp = []
best_templates = {}
continue_count = 0

with open(path, "r") as file:
    for line, gold, id, document in zip(file,ground_truths,ids,documents):
        data = json.loads(line)
        f1 = 0.0
        two_empty = False
        gold_empty = False
        current_f1s = []
        pred_data = []
        for template in data["pred_json"]:
            if template != ["ERROR"] and template != [["ERROR"]]:
                #TODO: Filtratu bakarrik incident_type dauketen template-ak

                completion, ground_truth = postprocess("TST1-MUC3-0001",template,gold)
                score_result = score(pred_data=completion, ref_data=ground_truth)
                current_f1 = score_result["iterx_muc_slot_f1"]
                if gold == []:
                    gold_empty = True
                if gold == [] and template == []:
                    current_f1 = 1.0
                if only_incident_template(template) and template == gold:
                    current_f1 = 1.0
                if only_incident_template(template) and gold == []:
                    current_f1 = 0.95
                if current_f1 > f1:
                    best_template = template
                    f1 = current_f1
                if f1 == 0 and template == []:
                    best_template = template
                    current_f1 += 0.001
                current_f1s.append(current_f1)
                pred_data.append((current_f1, template))

        pred_id = str(
            int(id.split("-")[0][-1]) * 10000
            + int(id.split("-")[-1]))
        #print(pred_id)
        pred_data.sort(key=lambda x: x[0], reverse=True)
        
        best_template = pred_data[0][1]
        if pred_id == "30565":
            idx = data["pred_json"].index(best_template)
            best_reasoning = data["pred_reasoning"][idx]
            print(best_reasoning)
            print(best_template)
            print(gold)

        best_templates[pred_id] = {"pred_templates": best_template, "gold_templates": gold}

values = score(pred_data=best_templates, ref_data=labels)
print("MAX F1: " + str(values["iterx_muc_slot_f1"]))