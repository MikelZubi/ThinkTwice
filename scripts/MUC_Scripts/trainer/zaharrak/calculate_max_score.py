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
    #Honek itten duna ya iñe
    predictions = load_predictions(pred_file=pred_file,
                                   dataset=dataset,
                                   file_type=file_type,
                                   normalize_role=normalize_role,
                                   remove_span_whitespace=remove_span_whitespace,
                                   scirex_merge_mentions=scirex_merge_mentions)
    #print(predictions)
    #print("\n\nSALTO\n\n")
    metric = load_metric(dataset_kind=dataset,
                         doc_path={str(pred_file): str(ref_file)},
                         ignore_no_template_doc=ignore_no_template_doc,
                         sanitize_special_chars=sanitize_special_chars,
                         scorer_type=scorer,
                         convert_doc_id=convert_doc_id)
    #print(metric.references)
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






pred_path = "perfect_dev.json"
pred_dict = {}
with open("multimuc/data/multimuc_v1.0/corrected/en/dev_simplified_preprocess.jsonl") as f:
    for line in f:
        data = json.loads(line)
        docid = str(
                int(data["docid"].split("-")[0][-1]) * 10000
                + int(data["docid"].split("-")[-1])
            )
        pred_dict[docid] = {}
        pred_dict[docid]["doctext"] = data["doctext"]
        pred_dict[docid]["gold_templates"] = data["templates"]

for docid in pred_dict.keys():
    post_templates = []
    for template in pred_dict[docid]["gold_templates"]["templates"]:
        post_processed = {}
        for key in template.keys():
            if key != "incident_type" and template[key] != []:
                post_processed[key]=[[elem] for elem in template[key]]
            else:
                post_processed[key]=template[key]
        post_templates.append(post_processed)
    pred_dict[docid]["pred_templates"] = post_templates
with open(pred_path, "w") as outfile:
    json.dump(pred_dict, outfile, indent=4,ensure_ascii=False)


gold_path = "multimuc/data/multimuc_v1.0/corrected/en/dev.jsonl"
pred_path = "perfect_dev.json"
results = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("Max score dev: " + str(results["iterx_muc_slot_f1"]))