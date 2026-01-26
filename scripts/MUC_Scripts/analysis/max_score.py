import sys
sys.path.append('multimuc/iterx/src')
from typing import OrderedDict, List, Union, Tuple, Optional, Callable, Dict

from iterx.metrics.muc.ceaf_rme import generate_scoring_structures, IterXTemplate, SCORER_CONSTRUCTOR
from iterx.metrics.muc.ceaf_rme import ScoreFunction

from pathlib import Path
from typing import Annotated
import typer
from iterx.metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric, \
    load_references, print_prediction_comparison
from iterx.metrics.muc.ceaf_rme import ScoreFunction
import os
import numpy as np
import json
from tqdm import tqdm

def smallest_template(all_templates):
    min_counter = float('inf')
    best_template = None
    for templates in all_templates:
        counter = 0
        for template in templates:
            counter += len(template["PerpInd"]) + len(template["PerpOrg"]) + len(template["Target"]) + len(template["Victim"]) + len(template["Weapon"])
        if counter < min_counter:
            min_counter = counter
            best_template = templates
    return best_template
        



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

def remove_errors(all_templates):
    return [template for template in all_templates if template != ["ERROR"] and template != [['ERROR']] and "ERROR" not in template]

def best_templates(entities_path,gold_path):
    labels = []
    with open(gold_path, "r") as file:
        for line in file:
            data = json.loads(line)
            corrected_id = data["docid"]
            label = {"docid": corrected_id, "templates": data["templates"], "doctext": data["doctext"]}
            labels.append(label)
    entities = []
    with open(entities_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            pred_templates = data["pred_json"]
            document = data["doctext"]
            docid = data["docid"]
            entities.append({"docid": docid, "templates": pred_templates, "doctext": document})
    
    #templates = list(filter(([]).__ne__, templates))
    max_entities = []
    for entity, label in tqdm(zip(entities, labels)):
        max_f1 = -1.0
        templates = entity["templates"]
        document = entity["doctext"]
        gold_template = label["templates"]
        templates = remove_errors(templates)
        if templates == []:
            max_entities.append({"docid": entity["docid"], "templates": [], "doctext": entity["doctext"]})
            continue
        cleaned_templates = [remove_only_incident_temp(template) for template in templates]
        #cleaned_templates = templates
        for template in cleaned_templates:
            post_template, post_label = postprocess("TST1-MUC3-0001", template, gold_template)
            score_result = score(pred_data=post_template, ref_data=post_label)
            current_f1 = score_result["iterx_muc_slot_f1"]
            if current_f1 > max_f1 or (max_f1 <= 0.0 and template == []):
                max_f1 = current_f1
                best_template = template
        if max_f1 == 0.0 and template != []:
            best_template = smallest_template(cleaned_templates)
        max_entities.append({"docid": entity["docid"], "templates": best_template, "doctext": entity["doctext"]})
    return max_entities, labels

def max_score(entities_path,gold_path):
    best_entities, labels = best_templates(entities_path,gold_path)
    predictions = {}
    for entity, label in zip(best_entities, labels):
        id = entity["docid"]
        pred_id = str(
            int(id.split("-")[0][-1]) * 10000
            + int(id.split("-")[-1]))
        predictions[pred_id] = {"pred_templates":entity["templates"],"gold_templates":[]}
    score_results = score(pred_data=predictions, ref_data=labels)
    f1_results = score_results["iterx_muc_slot_f1"]
    return f1_results
    


    