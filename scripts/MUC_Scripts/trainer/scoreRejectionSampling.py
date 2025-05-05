from typing import OrderedDict, List, Union, Tuple, Optional, Callable, Dict
import json
import sys
import numpy as np
import csv
sys.path.append('multimuc/iterx/src')
sys.path.append("class_data/")

from iterx.metrics.muc.ceaf_rme import ScoreFunction

from typing import Annotated
import typer
from iterx.metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric
from iterx.metrics.muc.ceaf_rme import ScoreFunction
import os
from MUC_Class_simplified import *
import argparse
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


parser = argparse.ArgumentParser(description='Arguments required for the scorer')
parser.add_argument('--split', dest='split', type=str)
parser.set_defaults(split="dev")
args = parser.parse_args()
split = args.split


#READ GOLD
gold_path = "multimuc/data/multimuc_v1.0/corrected/en/"+split+".jsonl"#IDATZI
ground_truths = []
ids = []
documents = []
labels = []
with open(gold_path, "r") as file:
    for line in file:
        data = json.loads(line)
        ground_truths.append(data["templates"])
        if split == "dev":
            ids.append(data["docid"])
            corrected_id = data["docid"]
        else:
            splited_ids = data["docid"].split("-")
            corrected_id = splited_ids[1] + "-" + splited_ids[0] + "-" + splited_ids[2]
            ids.append(corrected_id)
        documents.append(data["doctext"])
        label = {"docid": corrected_id, "templates": data["templates"]}
        labels.append(label)

completions = []
#paths = {"Reasoning": lambda x: "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/dev_Reasoning_"+str(x)+".jsonl", "StepReasoning": lambda x: "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/dev_StepReasoning_"+str(x)+".jsonl", "JSON": lambda x: "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/dev_JSON_"+str(x)+".jsonl"}

paths = {}
#Iterate files in rejectionSampling/dev
for file in os.listdir("rejectionSampling/"+split):
    if file.endswith(".jsonl"):
        #Extract the type and n from the filename
        file_type = "_".join(file.split("_")[:-1])
        #Add the path to the dictionary
        paths[file_type] = lambda x, ct=file_type: "rejectionSampling/"+split+"/" + ct + "_" + str(x) + ".jsonl"



header = ["Type","n","F1","Precision","Recall","STD","Mean"]
out_list = []
ns = [64]
stds = []
means = []
for key in paths:
    for n in ns:
        stds.clear()
        path = paths[key](n)
        best_f1s = []
        dis = 0
        best_templates = {}
        with open(path, "r") as file:
            for line, gold, id, document in zip(file,ground_truths,ids,documents):
                data = json.loads(line)
                f1 = 0.0
                two_empty = False
                gold_empty = False
                best_template = []
                current_f1s = []
                current_f1 = 0.0
                print(len(data["pred_json"]))
                for template in data["pred_json"]:
                    if template != ["ERROR"] and template != [["ERROR"]]:
                        completion, ground_truth = postprocess("TST1-MUC3-0001",template,gold)
                        score_result = score(pred_data=completion, ref_data=ground_truth)
                        current_f1 = score_result["iterx_muc_slot_f1"]
                        if gold == []:
                            gold_empty = True
                        if gold == [] and template == []:
                            current_f1 = 1.0
                        if current_f1 > f1:
                            best_template = template
                            f1 = current_f1
                        current_f1s.append(current_f1)
                best_f1s.append(f1)
                pred_id = str(
                        int(id.split("-")[0][-1]) * 10000
                        + int(id.split("-")[-1]))
                if n > 1:
                    std = np.std(current_f1s)
                    mean = np.mean(current_f1s)
                else:
                    std = 0
                    mean = current_f1
                if not gold_empty:
                    means.append(mean)
                    stds.append(std)
                best_templates[pred_id] = {"pred_templates":best_template,"gold_templates":gold}
            values = score(pred_data=best_templates, ref_data=labels)
            precision = values["iterx_muc_slot_p"]
            recall = values["iterx_muc_slot_r"]
            f1 = values["iterx_muc_slot_f1"]
            std = np.mean(stds)
            mean = np.mean(means)
            print(mean)
            out_list.append([key,n,f1,precision,recall,std,mean])

out_path = "rejectionSampling/"+split+"/scores.csv"
with open(out_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(out_list)