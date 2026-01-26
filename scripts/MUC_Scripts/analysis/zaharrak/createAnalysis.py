from typing import OrderedDict, List, Union, Tuple, Optional, Callable, Dict
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


#READ GOLD
gold_path = "multimuc/data/multimuc_v1.0/corrected/en/dev.jsonl"#IDATZI
ground_truths = []
ids = []
documents = []
labels = []
with open(gold_path, "r") as file:
    for line in file:
        data = json.loads(line)
        ground_truths.append(data["templates"])
        ids.append(data["docid"])
        documents.append(data["doctext"])
        label = {"docid": data["docid"], "templates": data["templates"]}
        labels.append(label)

completions = []
path = "multimuc/data/multimuc_v1.0/corrected/en/rejectionSampling/dev_Reasoning_32.jsonl"#IDATZI
best_f1s = []
dis = 0
all_data = {}
stds = []
best_templates = {}
bad_empty = None
good_empty = None
emptys = []
with open(path, "r") as file:
    for line, gold, id, document in zip(file,ground_truths,ids,documents):
        data = json.loads(line)
        f1 = 0.0
        two_empty = False
        gold_empty = False
        pred_data = []
        best_template = []
        incorrect_templates = []
        incorrect_templates_reasoning = []
        for template,reasoning in zip(data["pred_json"],data["pred_reasoning"]):
            if template != ["ERROR"]:
                completion, ground_truth = postprocess("TST1-MUC3-0001",template,gold)
                score_result = score(pred_data=completion, ref_data=ground_truth)
                current_f1 = score_result["iterx_muc_slot_f1"]
                if gold == []:
                    gold_empty = True
                if gold == [] and template == [] and not two_empty:
                    two_empty = True
                    correct_template = template
                    correct_reasoning = reasoning
                    dis += 1
                elif gold == [] and template != []:
                    incorrect_templates.append(template)
                    incorrect_templates_reasoning.append(reasoning)
                if current_f1 > f1:
                    best_template = template
                    f1 = current_f1
                pred_data.append((current_f1,template,reasoning))
        best_f1s.append(f1)
        pred_id = str(
                int(id.split("-")[0][-1]) * 10000
                + int(id.split("-")[-1]))
        best_templates[pred_id] = {"pred_templates":best_template,"gold_templates":gold}
        pred_data.sort(key=lambda x: x[0])
        reasoning = [x[2] for x in pred_data]
        sorted_templates = [x[1] for x in pred_data]
        current_f1s = [x[0] for x in pred_data]
        mean = np.mean(current_f1s)
        std = np.std(current_f1s)
        if not gold_empty:
            stds.append((id,std,mean,f1))
        if gold_empty and len(incorrect_templates) > 0 and two_empty:
            resumed_reasoning = incorrect_templates_reasoning[:2]
            resumed_reasoning.append(correct_reasoning)
            resumed_templates = incorrect_templates[:2]
            resumed_templates.append(correct_template)
            empty = {"document":document, "reasonings":resumed_reasoning, "gold_template":gold, "pred_templates":resumed_templates}
            emptys.append(empty)
            continue 

        resumed_templates = [sorted_templates[0], sorted_templates[len(sorted_templates)//2], sorted_templates[-1]]
        resumed_reasoning = [reasoning[0], reasoning[len(reasoning)//2], reasoning[-1]]
        resumed_f1s = [current_f1s[0], current_f1s[len(current_f1s)//2], current_f1s[-1]]        
        all_data[id] = {"document":document, "reasonings":resumed_reasoning, "gold_template":gold, "pred_templates":resumed_templates, "f1s":resumed_f1s, "mean":mean, "std":std}

print(sum(best_f1s)/(len(best_f1s)-dis))
print(score(pred_data=best_templates, ref_data=labels))
stds.sort(key=lambda x: x[1],reverse=True)
errorAnalysis = {}
errorAnalysis["empty_0"] = emptys[0]
errorAnalysis["empty_1"] = emptys[1]
errorAnalysis["std_high_1"] = all_data[stds[0][0]]
errorAnalysis["std_high_2"] = all_data[stds[1][0]]
errorAnalysis["std_low_1"] = all_data[stds[-1][0]]
errorAnalysis["std_low_2"] = all_data[stds[-2][0]]
print(stds)
with open ("errorAnalysis.json","w") as file:
    json.dump(errorAnalysis, file, indent=4, ensure_ascii=False)


