import sys
sys.path.append('BETTER_Scorer/')
sys.path.append("scripts/BETTER_Scripts/results")
import json
from postprocess_BETTER import dict_postprocessed_data
from tqdm import tqdm
import score
from lib.bp import BPDocument

def obtain_best_per_entry(entries_path, gold_path):
    with open(gold_path, 'r') as f:
        gold_data = json.load(f)
    with open(entries_path, 'r') as f:
        entries_data = []
        for line in f:
            entries_data.append(json.loads(line))
    max_post_templates = []
    for entry in tqdm(entries_data):
        #CREATE GOLD ENTRY DOCUMENT
        entry_id = entry['docid']
        document = entry["doctext"]
        gold_entry = gold_data["entries"][entry_id]
        gold_entry_document = BPDocument.from_dict({"corpus-id": gold_data["corpus-id"],
        "entries":{entry_id: gold_entry},
        "format-type": gold_data["format-type"],
        "format-version": gold_data["format-version"]})

        #CLEAN TEMPLATES
        no_error_templates = [template for template in entry['templates'] if "ERROR" not in template and "ERROR" not in template[0]]
        post_templates = []
        if no_error_templates == []:
            max_post_template = ["ERROR"]
        cleaned_templates = []
        for template in no_error_templates:
            if template not in cleaned_templates:
                cleaned_templates.append(template)

        #CALCULATE BEST TEMPLATE
        max_score = -1.0
        max_post_template = []
        for template in cleaned_templates:
            try:
                line_template = [{"docid": entry_id, "doctext": document, "templates": template}] 
                post_process_template, error_count = dict_postprocessed_data(line_template, simplified=True, dict_input=True)
                template_data = BPDocument.from_dict(post_process_template)
                if not template_data.is_valid():
                    continue
                score_granular, _ = score.score_granular(template_data, gold_entry_document, no_validation=True)
                current_score = score_granular.combined_score
                if current_score > max_score:
                    max_score = current_score
                    max_post_template = template
            except Exception as e:
                continue
        max_post_templates.append(max_post_template)
    return max_post_templates


def max_score(entries_path, gold_path):
    best_templates = obtain_best_per_entry(entries_path, gold_path)
    with open(entries_path, 'r') as f:
        entries_data = []
        for line in f:
            entries_data.append(json.loads(line))
    line_templates = []
    for entry, best_template in zip(entries_data, best_templates):
        entry_id = entry['docid']
        document = entry["doctext"]
        template = best_template
        line_template = {"docid": entry_id, "doctext": document, "templates": template}
        line_templates.append(line_template)
    post_process_template, error_count = dict_postprocessed_data(line_templates, simplified=True, dict_input=True)
    template_data = BPDocument.from_dict(post_process_template)
    true_data = BPDocument.from_json(gold_path)
    score_granular, _ = score.score_granular(template_data, true_data, no_validation=True)
    max_score = score_granular.combined_score
    return max_score
    


    