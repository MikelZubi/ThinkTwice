import pylcs
import sys
import os
import json
import copy as cp
from tqdm import tqdm


def detect_largest_coincidency(document, span_string):
    res = pylcs.lcs_string_idx(span_string, document)
    result = ''.join([document[i] for i in res if i != -1])
    start = 0
    if result != "":
        lag_res = [i for i in res if i != -1]
        start = lag_res[0]
    return result, start


path = sys.argv[1]
outpath = path + "_post/"
path += "/"
few_types = ["first-few/", "random-few/"]
languages = ["ar/", "en/", "fa/", "ko/", "ru/","zh/"]
#For gutxi batzuk
for few_type in few_types:
    for language in tqdm(languages):
        for file in os.listdir(path+few_type+language):
            new_data_list = []
            with open(path+few_type+language+file,"r") as f:
                data = json.load(f)
                new_data = cp.deepcopy(data)
                for id in data:
                    document = data[id]["doctext"]
                    for w, template in enumerate(data[id]["pred_templates"]):
                        for key in template:
                            if key != "incident_type":
                                new_data[id]["pred_templates"][w][key] = []
                                for i, _ in enumerate(template[key]):
                                    helper = []
                                    for j, corref in enumerate(template[key][i]):
                                        new_corref, _ = detect_largest_coincidency(document, corref)
                                        if new_corref != "":
                                            helper.append(new_corref)
                                    if helper != []:
                                        new_data[id]["pred_templates"][w][key].append(helper)
                                if new_data[id]["pred_templates"][w][key] == []:
                                    del new_data[id]["pred_templates"][w][key]
                                        
                                        
            os.makedirs(outpath+few_type+language, exist_ok=True)
            with open(outpath+few_type+language+file, 'w') as f:
                json.dump(new_data, f, indent=4, ensure_ascii=False)