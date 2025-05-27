
import sys
from typing import OrderedDict, List, Union, Tuple, Optional, Callable, Dict
import json

sys.path.append('multimuc/iterx/src')

from iterx.metrics.muc.ceaf_rme import ScoreFunction

from typing import Annotated
import typer
from iterx.metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric, \
    load_references, print_prediction_comparison
from iterx.metrics.muc.ceaf_rme import ScoreFunction



class FormatError(Exception):
    pass


def empty_template(pred,label):
    label_json = json.loads(label)
    try:
        pred_json = json.loads(pred)
        if type(pred_json) != dict:
            raise FormatError("The prediction is not a dictionary.")
    except json.JSONDecodeError:
        raise FormatError("The prediction is not a valid JSON")

    return len(pred_json["templates"]) == 0 and len(label_json) == 0

def postprocess(ids,preds,labels):
    post_preds = {}
    post_labels = []
    for id, pred, label in zip(ids,preds,labels):
        #print("PREDICTIONS!\n\n")
        #print(pred)        
        #print("Labels!\n\n")
        #print(label)
        pred_json = json.loads(pred)
        pred_json = {"pred_templates": pred_json["templates"]}
        print("JSONDecodeError")
        pred_json = {"pred_templates": []}
        label_json = json.loads(label)
        pred_id = str(
                int(id.split("-")[0][-1]) * 10000
                + int(id.split("-")[-1]))
        post_preds[pred_id] = pred_json
        post_labels.append({"docid": id, "templates": label_json["templates"]})
    return post_preds, post_labels

def postprocess_GRPO(id,pred,label):
    post_preds = {}
    post_labels = []
    pred_json = '{"templates":' + pred.split('{"templates":')[1]
    label_json = json.loads(label)
    pred_json = json.loads(pred)
    post_templates = []
    try:
        for template in pred_json["templates"]:
            post_processed = {}
            for key in template.keys():
                if key != "incident_type" and template[key] != []:
                    post_processed[key]=[[elem] for elem in template[key]]
                else:
                    post_processed[key]=template[key]
            post_templates.append(post_processed)
    except:
        raise FormatError("The prediction is not a valid JSON")
    pred_json = {"pred_templates": post_templates, "gold_templates": label_json}
    pred_id = str(
                int(id.split("-")[0][-1]) * 10000
                + int(id.split("-")[-1]))
    post_preds[pred_id] = pred_json
    post_labels.append({"docid": id, "templates": label_json})
    return post_preds, post_labels    

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