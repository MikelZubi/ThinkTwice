import json
import sys
sys.path.append('scripts/MUC_Scripts/scorer')
from scorers import line_scorer
from scipy.stats import spearmanr
import argparse
import os
from tqdm import tqdm

def remove_errors(all_templates):
    return [template for template in all_templates if ["ERROR"]  != template and [["ERROR"]] != template and "ERROR" not in template]

def measure_precision(str_template):
    error_measure = 1.0
    for template in str_template:
        for key in template.keys():
            if key != "incident_type":
                error_measure -= len(template[key]) * 0.025
    return error_measure


def calculate_correlations(reward_path,sampling_path,test_path,remove_repeated=False):
    with open(reward_path, 'r', encoding='utf-8') as f:
        reward_data = [json.loads(line) for line in f]
    with open(sampling_path, 'r', encoding='utf-8') as f:
        sampling_data = [json.loads(line) for line in f]
    with open(test_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    outputs = []
    for reward_item, sampling_item, test_item in zip(reward_data, sampling_data, test_data):
        docid = reward_item["docid"]
        score_dict = reward_item["score_dict"]
        gold = test_item["templates"]
        sampling_templates = sampling_item["pred_json"]
        sampling_templates = remove_errors(sampling_templates)
        if remove_repeated:
            sampling_templates = list(set([json.dumps(template, ensure_ascii=False) for template in sampling_templates]))
            sampling_templates = [json.loads(template) for template in sampling_templates]
        predict_scores = []
        real_scores = []

        for template in sampling_templates:
            str_template = json.dumps(template, ensure_ascii=False)
            if "ERROR" in template:
                continue
            predict_score = score_dict[str_template]
            if gold == []:
                real_score = measure_precision(template)
            else:
                real_score = line_scorer(template, gold)["iterx_muc_slot_f1"]
            predict_scores.append(predict_score)
            real_scores.append(real_score)
        correlation, p_value = spearmanr(predict_scores, real_scores)
        output = {
            "docid": docid,
            "predict_scores": predict_scores,
            "real_scores": real_scores,
            "correlation": correlation,
            "p_value": p_value
        }
        outputs.append(output)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Spearman correlation')
    parser.add_argument('--language', type=str, required=True, help='Path to the input file')
    parser.add_argument('--split', type=str, default="test", help='Dataset split to use (e.g., test)')
    parser.add_argument('--remove-repeated', dest="remove_repeated", action='store_true', default=False, help='Whether to remove repeated templates in sampling data')
    args = parser.parse_args()
    language = args.language
    split = args.split
    remove_repeated = args.remove_repeated
    input_path = f"results/MUC/zeroshot/{split}/{language}"   
    models = ["MUCQWEN_think", "MUCQWEN_nothink", "Qwen3-32B_think", "Qwen3-32B_nothink", "LlamaR1-70B", "MUCR1", "Llama3.3-70B", "MUCLLAMA"]
    for model in tqdm(models):
        reward_path = f"{input_path}/Reward/{model}-Reward_1.jsonl"
        sampling_path = f"{input_path}/{model}_64.jsonl"
        test_path = f"multimuc/data/multimuc_v1.0/corrected/{language}/{split}.jsonl"

        results = calculate_correlations(reward_path, sampling_path, test_path, remove_repeated=remove_repeated)
        if remove_repeated:
            output_path = f"results/MUC/zeroshot/analysis/{split}/{language}/{model}_reward_correlation_remove_repeated.jsonl"
        else:
            output_path = f"results/MUC/zeroshot/analysis/{split}/{language}/{model}_reward_correlation.jsonl"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    