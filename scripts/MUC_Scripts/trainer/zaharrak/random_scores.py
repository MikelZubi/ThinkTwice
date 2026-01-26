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
import random as rd
from tqdm import tqdm


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
    # Honek itten duna ya iñe
    predictions = load_predictions(pred_file=pred_data,
                                   dataset=dataset,
                                   file_type=file_type,
                                   normalize_role=normalize_role,
                                   remove_span_whitespace=remove_span_whitespace,
                                   scirex_merge_mentions=scirex_merge_mentions,
                                   compute_metrics=True)
    # print(predictions)
    # print("\n\nSALTO\n\n")
    metric = load_metric(dataset_kind=dataset,
                         doc_path={"dev": ref_data},
                         ignore_no_template_doc=ignore_no_template_doc,
                         sanitize_special_chars=sanitize_special_chars,
                         scorer_type=scorer,
                         convert_doc_id=convert_doc_id,
                         compute_metrics=True)
    # print(metric.references)
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


def remove_errors(all_templates):
    return [template if "ERROR" not in template else [] for template in all_templates]


def select_rd_templates(entities_path):
    entities = []
    with open(entities_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            templates = data["pred_json"]
            document = data["doctext"]
            docid = data["docid"]
            entities.append({"docid": docid, "templates": templates, "doctext": document})

    # templates = list(filter(([]).__ne__, templates))
    selected_entities = []
    for entity in entities:
        templates = entity["templates"]
        document = entity["doctext"]
        templates = remove_errors(templates)
        if templates == []:
            selected_entities.append({"docid": entity["docid"], "templates": [], "doctext": entity["doctext"]})
            continue
        selected_template = rd.choice(templates)
        selected_entities.append(
            {"docid": entity["docid"], "templates": selected_template, "doctext": entity["doctext"]})
    return selected_entities


def random_scores(entities_path, gold_path, n=100):
    rd.seed(42)
    labels = []
    with open(gold_path, "r") as file:
        for line in file:
            data = json.loads(line)
            corrected_id = data["docid"]
            label = {"docid": corrected_id, "templates": data["templates"], "doctext": data["doctext"]}
            labels.append(label)
    all_scores = []
    for i in tqdm(range(n)):
        selected_entities = select_rd_templates(entities_path)
        predictions = {}
        for entity, label in zip(selected_entities, labels):
            docid = entity["docid"]
            pred_id = str(
                int(docid.split("-")[0][-1]) * 10000
                + int(docid.split("-")[-1]))
            predictions[pred_id] = {"pred_templates": entity["templates"], "gold_templates": []}
        score_results = score(pred_data=predictions, ref_data=labels)
        f1_results = score_results["iterx_muc_slot_f1"]
        all_scores.append(f1_results)
    return all_scores



