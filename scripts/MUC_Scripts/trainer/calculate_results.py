import sys
import json
import copy as cp
from typing import OrderedDict, List, Union, Tuple, Optional, Callable, Dict

sys.path.append('iterx')

from metrics.muc.ceaf_rme import generate_scoring_structures, IterXTemplate, SCORER_CONSTRUCTOR
from metrics.muc.ceaf_rme import ScoreFunction


from pathlib import Path
from typing import Annotated
import typer
from metrics.ceaf_rme_cmd_utils import DatasetKind, PredictionFileType, load_predictions, load_metric, \
    load_references, print_prediction_comparison
from metrics.muc.ceaf_rme import ScoreFunction
import os
import csv
import argparse
import glob



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

#PROBA
'''
gold_path = "proba_gold.jsonl"
pred_path = "proba_pred.json"
results1 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
pred_path = "multimuc/predictions/gtt/tgt_man/ar/test_preds.json"
gold_path = "multimuc/data/multimuc_v1.0/corrected/ar/test.jsonl"
results1 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results PROBA2: " + str(results1["iterx_muc_slot_f1"]))
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
#pred_path = "predictions/MUC_simplified_SFT_JSONR1/en/greedy.json"
#results1 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
#print("results JSON: " + str(results1["iterx_muc_slot_f1"]))
#pred_path = "predictions/MUC_simplified_SFT_ReasoningR1/en/greedy.json"
#results2 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
#print("results Reasoning: " + str(results2["iterx_muc_slot_f1"]))
#pred_path = "predictions/MUC_simplified_SFT_Natural_ReasoningR1/en/greedy.json"
#results3 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
#print("results Natural Reasoning: " + str(results3["iterx_muc_slot_f1"]))


#GRPO


pred_path = "predictions/MUC_simplified_GRPO_JSON/en/greedy.json"
results4 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results GRPO JSON: " + str(results4["iterx_muc_slot_f1"]))
pred_path = "predictions/MUC_simplified_GRPO_Natural_Reasoning/en/greedy.json"
results4 = score(pred_path, gold_path, DatasetKind.MUC, file_type=PredictionFileType.GTT)
print("results GRPO Natural Reasoning: " + str(results4["iterx_muc_slot_f1"]))
'''

def calculate_scores_for_directory(read,split,scorer,certainly,voterf1, maxlen, reward):
    gold_path = "multimuc/data/multimuc_v1.0/corrected/en/"+split+".jsonl"#IDATZI
    ground_truths = []
    ids = []
    documents = []
    labels = []
    with open(gold_path, "r") as file:
        for line in file:
            data = json.loads(line)
            ground_truths.append(data["templates"])
            ids.append(data["docid"])
            corrected_id = data["docid"]
            documents.append(data["doctext"])
            label = {"docid": corrected_id, "templates": data["templates"]}
            labels.append(label)
    """
    Iterate through each file in rejectionSampling/dev/5 directory,
    calculate scores, and save results to a CSV file.
    """
    # Define paths
    prediction_dir = read
    output_csv = read +".csv"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Find all prediction files
    if not scorer and not certainly and not voterf1 and not maxlen and not reward:
        prediction_files = glob.glob(os.path.join(prediction_dir, "*_1.jsonl"))
    else:
        prediction_files = glob.glob(os.path.join(prediction_dir, "*.jsonl"))
    print(prediction_files)
    # Prepare CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['file', 'precision', 'recall', 'f1', 'num_errors']
        #fieldnames = ['file', 'precision', 'recall', 'f1', "num_errors", "f1_scorer", "precision_scorer", "recall_scorer"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each prediction file
        for pred_file in prediction_files:
            # Skip files that don't end with _1.jsonl
            file_name = os.path.basename(pred_file)
            print(f"Processing {file_name}...")
            predictions = {}
            #predictions_scorer = {}
            count = 0
            with open(pred_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    # Extract the ID from the JSON data
                    id = data["docid"]
                    pred_id = str(
                        int(id.split("-")[0][-1]) * 10000
                        + int(id.split("-")[-1]))
                    if scorer:
                        template = data["pred_json_scorer"]
                    elif voterf1:
                        template = data["pred_json_scorerf1"]
                    elif certainly:
                        template = data["pred_certainly_json"]
                    elif maxlen:
                        template = data["pred_json_maxlen"]
                    elif reward:
                        template = data["pred_json_reward"]
                    else:
                        template = data["pred_json"]
                    if template == ["ERROR"] or template == [["ERROR"]]:
                        count += 1
                        template = []
                    predictions[pred_id] = {"pred_templates":template,"gold_templates":data["templates"]}
            # Calculate scores
            results = score(
                pred_data=predictions,
                ref_data=labels
            )
            #results_scorer = score(pred_data=predictions_scorer, ref_data=labels)

            # Extract key metrics
            precision = results["iterx_muc_slot_p"]
            recall = results["iterx_muc_slot_r"]
            f1 = results["iterx_muc_slot_f1"]
            row_dict = {
                'file': file_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'num_errors': count
            }
            '''
            # Update the fieldnames list to include 'num_errors'
            count_w = 0
            file_w_names = file_name.split("_1.jsonl")[0] + "_1_W*"
            prediction_file_w = glob.glob(os.path.join(prediction_dir, file_w_names))
            for file_w in prediction_file_w:            
                with open(file_w, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        # Extract the ID from the JSON data
                        id = data["docid"]
                        pred_id = str(
                            int(id.split("-")[0][-1]) * 10000
                            + int(id.split("-")[-1]))
                        template = data["pred_json"] #TODO: Kendu
                        if template == ["ERROR"] or template == [["ERROR"]]:
                            count_w += 1
                            template = []
                        predictions[pred_id] = {"pred_templates":template,"gold_templates":data["templates"]}
                # Calculate scores
                results_w = score(
                    pred_data=predictions,
                    ref_data=labels
                )
                f1_w = results_w["iterx_muc_slot_f1"]
                file_w_count = file_w.split("_W")[-1][0]
                w_name = "f1w" + file_w_count
                row_dict[w_name] = f1_w
                if w_name not in writer.fieldnames:
                    writer.fieldnames.append(w_name)
                
                if "num_errors_w" not in writer.fieldnames:
                    writer.fieldnames.append("num_errors_w")
                row_dict['num_errors_w'] = count_w
            '''
                
            # Write to CSV including the error count
            writer.writerow(row_dict)
            
            
            print(f"Completed {file_name}: F1={round(f1, 4)}")
            #print(f"Completed {file_name}: {w_name}={round(f1_w, 4)}")
            print(f"Completed {file_name}: num_errors={count}")
    
    print(f"All scores saved to {output_csv}")

# Run the function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate scores for prediction files in a directory')
    parser.add_argument('--read', type=str, default='results', help='read path and output csv fiename') 
    parser.add_argument('--split', type=str, default='dev', help='split to use for scoring')   
    parser.add_argument('--voter', action='store_true', default=False, help='Use pred_json_scorer instead of pred_json')
    parser.add_argument('--voterf1', action='store_true', default=False, help='Use pred_json_scorer instead of pred_json')
    parser.add_argument('--certainly', action='store_true', default=False, help='Use pred_certainly_json instead of pred_json')
    parser.add_argument('--maxlen', action='store_true', default=False, help='Use pred_json_maxlen instead of pred_json')
    parser.add_argument('--reward', action='store_true', default=False, help='Use pred_json_reward instead of pred_json')

    args = parser.parse_args()
    calculate_scores_for_directory(args.read, args.split, args.voter, args.certainly, args.voterf1, args.maxlen, args.reward)

