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
parser.add_argument('--n', dest='n', type=int,
                    help='The number of best templates to select.')
parser.add_argument("--iter", dest="iter", type=str)
parser.add_argument("--read", dest="read", type=str)
parser.add_argument("--obtain-reasoning", dest="obtain_reasoning", action="store_true",)
parser.set_defaults(obtain_reasoning=False)
parser.set_defaults(n=32)
parser.set_defaults(iter=1)
n = parser.parse_args().n
iteration = parser.parse_args().iter
read = parser.parse_args().read
obtain_reasoning = parser.parse_args().obtain_reasoning
#READ GOLD
gold_path = "multimuc/data/multimuc_v1.0/corrected/en/train.jsonl"#IDATZI
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
        doctext = " ".join(data["doctext"].split())
        documents.append(doctext)
        label = {"docid": new_id, "templates": data["templates"]}
        labels.append(label)

completions = []
out_list = []
path = "rejectionSampling/train/"+iteration+"/"+read #TODO
best_f1s = []
dis = 0
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
        for template,reasoning in zip(data["pred_json"],data["pred_reasoning"]):
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
                pred_data.append((current_f1, template, reasoning))

        pred_id = str(
            int(id.split("-")[0][-1]) * 10000
            + int(id.split("-")[-1]))
        print(pred_id)
        pred_data.sort(key=lambda x: x[0], reverse=True)
        best_template = pred_data[0][1]

        best_templates[pred_id] = {"pred_templates": best_template, "gold_templates": gold}
        selected_values = pred_data[:n]
        if not obtain_reasoning:
            for _, template, reasoning in selected_values:
                if template != ["ERROR"] and template != [["ERROR"]]:
                    post_template = simplify_template(template)
                    outputs.append({"docid": id, "completion": "<think>\n" + reasoning + "</THINK_TOKENA>" + post_template, "doctext": document})
            if n == 1:
                for _, template, _ in selected_values:
                    if template != ["ERROR"] and template != [["ERROR"]]:
                        post_template = simplify_template(template)
                        post_template = json.loads(post_template)
                        print("inserted")
                        out_onlytemp.append({"docid": id, "templates": post_template, "doctext": document})
        else:
            selected_values = pred_data
            reasonings = []
            for _, template, reasoning in selected_values:
                if template != ["ERROR"] and template != [["ERROR"]] and reasoning not in reasonings:
                    post_template = simplify_template(template)
                    outputs.append({"docid": id, "reasoning": "<think>\n" + reasoning + "</THINK_TOKENA>", "template": post_template, "doctext": document})
                    reasonings.append(reasoning)
values = score(pred_data=best_templates, ref_data=labels)
print("MAX F1: " + str(values["iterx_muc_slot_f1"]))
if not obtain_reasoning:
    out_path = "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/train_best"+str(n)+".jsonl"
    with open(out_path, 'w') as file:
        for line in outputs:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")

    if n == 1:
        out_path = "multimuc/data/multimuc_v1.0/corrected/en/sampled_template_simplified_preprocess.jsonl"
        with open(out_path, 'w') as file:
            for line in out_onlytemp:
                file.write(json.dumps(line, ensure_ascii=False) + "\n")
else:
    out_path = "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/train_obtain_reasoning.jsonl"
    with open(out_path, 'w') as file:
        for line in outputs:
            file.write(json.dumps(line, ensure_ascii=False) + "\n")
