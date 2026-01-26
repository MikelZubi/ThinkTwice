import sys
sys.path.append('BETTER_Scorer/')
sys.path.append("scripts/BETTER_Scripts/results")
from postprocess_BETTER import dict_postprocessed_data
import json
from tqdm import tqdm
import score
from lib.bp import BPDocument
import random as rd

def obtain_randoms_per_entry(entries_path):
    with open(entries_path, 'r') as f:
        entries_data = []
        for line in f:
            entries_data.append(json.loads(line))
    all_selected_templates = []
    for entry in entries_data:

        #CLEAN TEMPLATES
        no_error_templates = [template for template in entry['templates'] if "ERROR" not in template and "ERROR" not in template[0]]
        if no_error_templates == []:
            selected_template = ["ERROR"]
        else:
            selected_template = rd.choice(no_error_templates)
        all_selected_templates.append(selected_template)
    return all_selected_templates


def random_scores(entries_path, gold_path, n=100):
    rd.seed(42)
    true_data = BPDocument.from_json(gold_path)
    with open(entries_path, 'r') as f:
        entries_data = []
        for line in f:
            entries_data.append(json.loads(line))
    all_scores = []
    for i in tqdm(range(n)):
        all_selected_templates = obtain_randoms_per_entry(entries_path)
        line_templates = []
        for entry, selected_template in zip(entries_data, all_selected_templates):
            entry_id = entry['docid']
            document = entry["doctext"]
            template = selected_template
            line_template = {"docid": entry_id, "doctext": document, "templates": template}
            line_templates.append(line_template)
        post_process_template, error_count = dict_postprocessed_data(line_templates, simplified=True, dict_input=True)
        template_data = BPDocument.from_dict(post_process_template)
        score_granular, _ = score.score_granular(template_data, true_data, no_validation=True)
        current_score = score_granular.combined_score
        all_scores.append(current_score)
    return all_scores
    


    