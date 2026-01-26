import sys
import json
import os
sys.path.append('BETTER_Scorer/')
import score
from lib.bp import BPDocument
from postprocess_BETTER import write_postprocessed_data, dict_postprocessed_data
from tqdm import tqdm
import csv
import argparse
import glob


# Argument parser
parser = argparse.ArgumentParser(description='Arguments required to score BETTER predictions')
parser.add_argument('--read', dest='predictions_dir', type=str)
parser.add_argument('--str-templates', dest='str_template', action='store_true')
parser.add_argument("--split", dest='split', type=str, default='dev')
parser.set_defaults(str_template=False)
args = parser.parse_args()
predictions_dir = args.predictions_dir
split = args.split
str_template = args.str_template



true_file = f'phase2/phase2.granular.eng.{split}.json'
true_data = BPDocument.from_json(true_file)
true_data.is_valid()
# Create/open CSV file to store scores

fieldnames = ['file', 'template', 'slot', 'irrealis', 'time-attachment', 'better', "error_count"]
max_score = 0
csv_file = predictions_dir + "_scores.csv"
with open(csv_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    current_dir = predictions_dir
    prediction_files = glob.glob(os.path.join(predictions_dir, "*_1.jsonl"))
    num_errors = 0
    for file in prediction_files:
        #try:
        modelname = os.path.basename(file)
        output_path = file.replace(predictions_dir, predictions_dir + "/postprocessed/")
        output_file = os.path.join(output_path, file)[:-1] #remove l from jsonl
        print(file)
        #error_count = dict_postprocessed_data(file, output_file,simplified=True)
        #predict_data = BPDocument.from_json(output_file)
        postprocessed_data, error_count = dict_postprocessed_data(file, simplified=True,str_template=str_template)
        predict_data = BPDocument.from_dict(postprocessed_data)
        result, _ = score.score_granular(predict_data, true_data, no_validation=True)
        all_scores = {
                        'file': modelname,
                        'template': result.template_measures.f1,
                        'slot': result.slotmatch_measures.f1,
                        'irrealis': result.irrealis_measures.f1,
                        'time-attachment': result.time_attachment_measures.f1,
                        'better': result.combined_score,
                        'error_count': error_count
                    }
        if result.combined_score > max_score:
            max_score = result.combined_score
            max_file = file
        '''
        except Exception as e:
            all_scores = {
                            'modelname': modelname,
                            'template': 0,
                            'slot': 0,
                            'irrealis': 0,
                            'time-attachment': 0,
                            'better': 0,
                        }
            print("ERROR")
            print("Num errors: ", num_errors)
            num_errors += 1
            print("Error file: ", file)
            print(e)
        '''
        writer.writerow(all_scores)
