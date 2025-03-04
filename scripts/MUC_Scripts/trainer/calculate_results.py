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

gold_path = "multimuc/data/multimuc_v1.0/corrected/en/test.jsonl"
pred_path = "predictions/MUC_simplified_SFT_JSON/en/greedy.json"
results1 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results JSON: " + str(results1["iterx_muc_slot_f1"]))
pred_path = "predictions/MUC_simplified_SFT_Reasoning/en/greedy.json"
results2 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results Reasoning: " + str(results2["iterx_muc_slot_f1"]))
pred_path = "predictions/MUC_simplified_SFT_Natural_Reasoning/en/greedy.json"
results3 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results Natural Reasoning: " + str(results3["iterx_muc_slot_f1"]))
pred_path = "predictions/MUC_simplified_GRPO_JSON/en/greedy.json"
results4 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results GRPO JSON: " + str(results4["iterx_muc_slot_f1"]))
pred_path = "predictions/MUC_simplified_GRPO_Natural_Reasoning/en/greedy.json"
results4 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results GRPO Natural Reasoning: " + str(results4["iterx_muc_slot_f1"]))

