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
ks = list(range(0,61))
posibles_tags = ["first-few/", "random-few/"]
languages = ["en", "ar", "fa", "ko", "ru", "zh"]

for tag in posibles_tags:
    for language in tqdm(languages):
        gold_file = "multimuc/data/multimuc_v1.0/corrected/" + language + "/dev.jsonl"
        all_results = []
        selected_ks = []

        #REE
        if sys.argv[1] == "REE":
            for k in ks:
                pred_file = "predictions/"+ tag + language + "/" + str(k) + "-shot_greedy.json"
                ## get pred and gold extracts
                preds = OrderedDict()
                golds = OrderedDict()
                if not os.path.exists(pred_file):
                    continue
                selected_ks.append(k)
                with open(pred_file, encoding="utf-8") as f:
                    out_dict = json.load(f)
                    for docid in out_dict:
                        preds[docid] = out_dict[docid]["pred_templates"]

                bad_gold_mentions = 0
                total_gold_mentions = 0
                with open(gold_file, encoding="utf-8") as f:
                    for line in f:
                        line = json.loads(line)
                        docid = str(
                            int(line["docid"].split("-")[0][-1]) * 10000
                            + int(line["docid"].split("-")[-1])
                        )
                        templates_raw = line["templates"]
                        templates = []
                        for template_raw in templates_raw:
                            template = OrderedDict()
                            for role, value in template_raw.items():
                                if role == "incident_type":
                                    template[role] = value
                                else:
                                    template[role] = []
                                    for entity_raw in value:
                                        entity = []
                                        for mention_offset_pair in entity_raw:
                                            # This will drop any mention annotated in the gold data
                                            # that does not also appear in the document text. This may
                                            # happen because of (e.g.) poor translations.
                                            if is_valid_mention(
                                                mention_offset_pair, line["doctext"]
                                            ):
                                                entity.append(mention_offset_pair[0])
                                            else:
                                                bad_gold_mentions += 1
                                            total_gold_mentions += 1
                                        if entity:
                                            template[role].append(entity)
                            if template not in templates:
                                templates.append(template)
                        golds[docid] = templates
                # we'd like to know what fraction of gold mentions are dropped
                # due to not appearing in the document text
                print("bad gold mentions: {}/{}".format(bad_gold_mentions, total_gold_mentions))

                all_keys = list(role for _, role in tag2role.items()) + ["micro_avg"]
                docids = []
                results = eval_tf(preds, golds, docids)
                all_results.append(results)
            
            if not os.path.exists("results/"+ tag + language):
                os.makedirs("results/"+ tag + language)
        
            with open("results/"+ tag + language + "/CEAF-REE-greedy.jsonl", "w", encoding="utf-8") as f:
                for i, result in enumerate(all_results):
                    info = {}
                    info["k"] = ks[i]
                    info["results"] = result
                    json.dump(info, f)
                    f.write("\n")

            with open("results/"+ tag + language + "/CEAF-REE-greedy.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(["ks"]+[key for key in all_results[0]])
                for i, result in enumerate(all_results):
                    writer.writerow([ks[i]]+[result[key]["f1"] for key in result])

        #RME

        all_results = []
        for k in ks:
            pred_file = "predictions/"+ tag + language + "/" + str(k) + "-shot_greedy.json"
            if not os.path.exists(pred_file):
                continue
            selected_ks.append(k)
            ## get pred and gold extracts
            pre_results = score(pred_file, gold_file, DatasetKind.MUC, file_type=PredictionFileType.GTT)
            results = {"p": pre_results["iterx_muc_slot_p"], "r": pre_results["iterx_muc_slot_r"], "f1": pre_results["iterx_muc_slot_f1"]}
            all_results.append(results)
        

        if not os.path.exists("results/"+ tag + language + "/"):
            os.makedirs("results/"+ tag + language + "/")
        with open("results/"+ tag + language + "/CEAF-RME-greedy.jsonl", "w", encoding="utf-8") as f:
            for i, result in enumerate(all_results):
                info = {}
                info["k"] = selected_ks[i]
                info["results"] = result
                json.dump(info, f)
                f.write("\n")

        with open("results/"+ tag + language + "/CEAF-RME-greedy.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["ks"]+[key for key in all_results[0]])
            for i, result in enumerate(all_results):
                writer.writerow([selected_ks[i]]+[result[key] for key in result])

