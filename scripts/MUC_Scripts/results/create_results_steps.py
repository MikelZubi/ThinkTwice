import sys
from collections import OrderedDict
import os
import json
import csv
from multimuc.scripts.eval import eval_tf, is_valid_mention
import os 
from tqdm import tqdm


sys.path.append("multimuc/iterx/src")

from pathlib import Path
from typing import Annotated
import typer
from iterx.metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric, \
    load_references, print_prediction_comparison
from iterx.metrics.muc.ceaf_rme import ScoreFunction

#RME Functions

def score(
        pred_file: Path,
        ref_file: Path,
        dataset: Annotated[DatasetKind, typer.Option()],
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
        )] = PredictionFileType.IterX,
        remove_span_whitespace: Annotated[bool, typer.Option(
			help='Whether to concatenate all the spans in a mention to form a single span.'
        )] = False,
):
    """Score a prediction file against a reference file."""
    normalize_role = True if dataset == DatasetKind.MUC else False
    convert_doc_id = True if file_type == PredictionFileType.GTT and dataset == DatasetKind.MUC else False
    # SciREX postprocessing should have been moved out of the scorer.
    predictions = load_predictions(pred_file=pred_file,
                                   dataset=dataset,
                                   file_type=file_type,
                                   normalize_role=normalize_role,
                                   remove_span_whitespace=remove_span_whitespace,
                                   scirex_merge_mentions=scirex_merge_mentions)
    metric = load_metric(dataset_kind=dataset,
                         doc_path={str(pred_file): str(ref_file)},
                         ignore_no_template_doc=ignore_no_template_doc,
                         sanitize_special_chars=sanitize_special_chars,
                         scorer_type=scorer,
                         convert_doc_id=convert_doc_id)

    metric(
        predictions=predictions,
        pred_src_file=str(pred_file),
        dedup=False,
        cluster_substr=False,
        normalize_role=normalize_role
    )
    results = metric.get_metric(reset=True)
    for k, v in results.items():
        print(f"{k}: {round(v, 4)}")
    return results 



tag2role = OrderedDict(
    {
        "incident_type": "incident_type",
        "perp_individual_id": "PerpInd",
        "perp_organization_id": "PerpOrg",
        "phys_tgt_id": "Target",
        "hum_tgt_name": "Victim",
        "incident_instrument_id": "Weapon",
    }
)
all_results = []
selected_ks = list(range(0,61))
for k in range(61):
    gold_file = "multimuc/data/multimuc_v1.0/corrected/en/dev.jsonl"
    pred_file = "predictions/predictions_MUC_simplified_steps/first-few/en/"+str(k)+"-shot_greedy.json"
    pre_results = score(pred_file, gold_file, DatasetKind.MUC, file_type=PredictionFileType.GTT)
    results = {"p": pre_results["iterx_muc_slot_p"], "r": pre_results["iterx_muc_slot_r"], "f1": pre_results["iterx_muc_slot_f1"]}
    all_results.append(results)

with open("results/steps-MUC-RME-greedy.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["ks"]+[key for key in all_results[0]])
    for i, result in enumerate(all_results):
        writer.writerow([selected_ks[i]]+[result[key] for key in result])

all_results = []
selected_ks = list(range(0,61))
for k in range(61):
    gold_file = "multimuc/data/multimuc_v1.0/corrected/en/dev.jsonl"
    pred_file = "predictions/predictions_MUC_simplified_steps_70B/first-few/en/"+str(k)+"-shot_greedy.json"
    pre_results = score(pred_file, gold_file, DatasetKind.MUC, file_type=PredictionFileType.GTT)
    results = {"p": pre_results["iterx_muc_slot_p"], "r": pre_results["iterx_muc_slot_r"], "f1": pre_results["iterx_muc_slot_f1"]}
    all_results.append(results)


with open("results/steps-MUC-RME-greedy_70B.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["ks"]+[key for key in all_results[0]])
    for i, result in enumerate(all_results):
        writer.writerow([selected_ks[i]]+[result[key] for key in result])

all_results = []
selected_ks = list(range(0,61))
for k in range(61):
    gold_file = "multimuc/data/multimuc_v1.0/corrected/en/dev.jsonl"
    pred_file = "predictions/predictions_MUC_simplified_70B/first-few/en/"+str(k)+"-shot_greedy.json"
    pre_results = score(pred_file, gold_file, DatasetKind.MUC, file_type=PredictionFileType.GTT)
    results = {"p": pre_results["iterx_muc_slot_p"], "r": pre_results["iterx_muc_slot_r"], "f1": pre_results["iterx_muc_slot_f1"]}
    all_results.append(results)



os.makedirs("results/results_MUC_simplified_70B/first-few/en", exist_ok=True)
with open("results/results_MUC_simplified_70B/first-few/en/CEAF-RME-greedy.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["ks"]+[key for key in all_results[0]])
    for i, result in enumerate(all_results):
        writer.writerow([selected_ks[i]]+[result[key] for key in result])


selected_ks = [0,1,5,10,15,20,25,30,35,40,45,50,55,60]
gold_file = "multimuc/data/multimuc_v1.0/corrected/en/dev.jsonl"
all_results = []
for k in selected_ks:
    print(k)
    pred_file = "predictions/predictions_MUC_simplified_steps_deepseek/first-few/en/"+str(k)+"-shot_greedy.json"
    pre_results = score(pred_file, gold_file, DatasetKind.MUC, file_type=PredictionFileType.GTT)
    results = {"p": pre_results["iterx_muc_slot_p"], "r": pre_results["iterx_muc_slot_r"], "f1": pre_results["iterx_muc_slot_f1"]}
    all_results.append(results)

with open("results/steps-MUC-RME-greedy_deepseek.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["ks"]+[key for key in all_results[0]])
    for i, result in enumerate(all_results):
        print(i)
        writer.writerow([selected_ks[i]]+[result[key] for key in result])


selected_ks = [0,1,5,10,15,20,25,30,35,40,45,50,55,60]
gold_file = "multimuc/data/multimuc_v1.0/corrected/en/dev.jsonl"
all_results = []
for k in selected_ks:
    print(k)
    pred_file = "predictions/predictions_MUC_simplified_steps_deepseekV2/first-few/en/"+str(k)+"-shot_greedy.json"
    pre_results = score(pred_file, gold_file, DatasetKind.MUC, file_type=PredictionFileType.GTT)
    results = {"p": pre_results["iterx_muc_slot_p"], "r": pre_results["iterx_muc_slot_r"], "f1": pre_results["iterx_muc_slot_f1"]}
    all_results.append(results)

with open("results/steps-MUC-RME-greedy_deepseekV2.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["ks"]+[key for key in all_results[0]])
    for i, result in enumerate(all_results):
        print(i)
        writer.writerow([selected_ks[i]]+[result[key] for key in result])
