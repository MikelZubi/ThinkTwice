import sys
import json
import os
sys.path.append('BETTER_Scorer/')
import score
from lib.bp import BPDocument
from postprocess_BETTER import write_postprocessed_data
from tqdm import tqdm
import csv


true_file = 'phase2/phase2.granular.eng.dev.json'
true_data = BPDocument.from_json(true_file)
true_data.is_valid()
predictions_dir = sys.argv[1]
if os.path.exists("results_motxean.txt"):
    os.remove("results_motxean.txt")
# Create/open CSV file to store scores
csv_path = "results"+predictions_dir[11:]+"/"
fieldnames = ['shot', 'template', 'slot', 'irrealis', 'time-attachment', 'better']
max_score = 0
for option in ["first-few","random-few"]:
    csv_file = csv_path + option + ".csv"
    os.makedirs(csv_path, exist_ok=True)
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        current_dir = os.path.join(predictions_dir, option)
        num_errors = 0
        for file in tqdm(sorted(os.listdir(current_dir),key=lambda x: int(x.split('-')[0]))):
            try:
                num_shot = file.split('-')[0]
                predict_file = os.path.join(current_dir, file)
                output_path = current_dir.replace("predictions/predictions_BETTER", "predictions/postprocess_BETTER/postprocess_BETTER")
                output_file = os.path.join(output_path, file)[:-1] #remove l from jsonl
                simplified = False
                if "simplified" in predictions_dir:
                    simplified = True
                print(predict_file)
                write_postprocessed_data(predict_file, output_file,simplified=simplified)
                predict_data = BPDocument.from_json(output_file)
                predict_data.is_valid()
                result, _ = score.score_granular(predict_data, true_data,no_validation=True)
                all_scores = {
                                'shot': num_shot,
                                'template': result.template_measures.f1,
                                'slot': result.slotmatch_measures.f1,
                                'irrealis': result.irrealis_measures.f1,
                                'time-attachment': result.time_attachment_measures.f1,
                                'better': result.combined_score,
                            }
                if result.combined_score > max_score:
                    max_score = result.combined_score
                    max_file = file
            except Exception as e:
                all_scores = {
                                'shot': num_shot,
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
            writer.writerow(all_scores)
    with open("results_motxean.txt","a") as f:
        f.write("Max score: "+str(max_score)+"\n")
        f.write("Max file: "+str(max_file)+"\n")
        f.write("-----------------------------\n")
