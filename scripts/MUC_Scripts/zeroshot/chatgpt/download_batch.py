from openai import OpenAI
import json
import os
import argparse
from collections import defaultdict

import sys
sys.path.append('class_data')
sys.path.append('inference_library')

from MUC_Class_simplified import *
from utils import maxCommStr


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Download and process OpenAI batch results')

parser.add_argument('--batch-id-file', type=str, required=True)
parser.add_argument('--split', type=str)
parser.add_argument('--language', type=str)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--n', type=int, default=1)
parser.add_argument('--think', action='store_true')
parser.add_argument('--api-key', type=str, default=None)

parser.set_defaults(
    language='en',
    split='proba',
)

args = parser.parse_args()

split      = args.split
language   = args.language
path_write = args.out_dir
n          = args.n
think      = args.think


# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
api_key = args.api_key or os.environ.get('OPENAI_API_KEY')

if not api_key:
    raise ValueError('OPENAI_API_KEY not found')

client = OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Input path
# ---------------------------------------------------------------------------
path_read = (
    'multimuc/data/multimuc_v1.0/corrected/'
    + language + '/' + split + '_simplified_preprocess.jsonl'
)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------
def process_output(json_text: str, doctext: str) -> list:
    post_templates = []
    lower_doc = doctext.lower()

    try:
        for template in json.loads(json_text)['templates']:

            post_processed = {}

            for key in template.keys():

                if key != 'incident_type' and template[key] != []:

                    post_processed[key] = []

                    for elem in template[key]:

                        lower_elem = elem.lower()

                        if lower_elem in lower_doc:
                            post_processed[key].append([lower_elem])

                        else:
                            commn_str = maxCommStr(lower_elem, lower_doc)

                            if commn_str:
                                if commn_str[0] == ' ':
                                    commn_str = commn_str[1:]

                                if commn_str[-1] == ' ':
                                    commn_str = commn_str[:-1]

                                post_processed[key].append([commn_str])

                else:
                    post_processed[key] = template[key]

            post_templates.append(post_processed)

    except Exception:
        post_templates.append('ERROR')

    return post_templates


# ---------------------------------------------------------------------------
# Retrieve batch
# ---------------------------------------------------------------------------
with open(args.batch_id_file, 'r') as f:
    batch_id = f.read().strip()

batch = client.batches.retrieve(batch_id)

if batch.status != 'completed':
    raise ValueError(f'Batch not completed. Current status: {batch.status}')


# ---------------------------------------------------------------------------
# Download output file
# ---------------------------------------------------------------------------
output_file_id = batch.output_file_id

content = client.files.content(output_file_id)

raw_output_path = 'batch_results.jsonl'

with open(raw_output_path, 'wb') as f:
    f.write(content.read())

print(f'Downloaded batch results to: {raw_output_path}')


# ---------------------------------------------------------------------------
# Load original dataset
# ---------------------------------------------------------------------------
pre_dicts = []

with open(path_read, 'r') as file:
    for line in file:
        pre_dicts.append(json.loads(line))


# ---------------------------------------------------------------------------
# Parse outputs
# ---------------------------------------------------------------------------
results = defaultdict(list)
reasonings = defaultdict(list)

with open(raw_output_path, 'r') as f:

    for line in f:

        item = json.loads(line)

        custom_id = item['custom_id']

        parts = custom_id.split('_')
        doc_idx = int(parts[1])


        # ---------------------------------------------------------------
        # Responses API path
        # ---------------------------------------------------------------
        if think:

            body = item['response']['body']

            output_text = ''
            reasoning_summary = None

            for output_item in body['output']:

                if output_item['type'] == 'reasoning':

                    if output_item.get('summary'):
                        reasoning_summary = ' '.join([
                            block['text']
                            for block in output_item['summary']
                            if 'text' in block
                        ])

                elif output_item['type'] == 'message':

                    for block in output_item['content']:
                        if 'text' in block:
                            output_text += block['text']

            results[doc_idx].append(output_text)
            reasonings[doc_idx].append(reasoning_summary)


        # ---------------------------------------------------------------
        # Chat Completions path
        # ---------------------------------------------------------------
        else:

            output_text = (
                item['response']['body']['choices'][0]
                ['message']['content']
            )

            results[doc_idx].append(output_text)


# ---------------------------------------------------------------------------
# Reconstruct predictions
# ---------------------------------------------------------------------------
for idx, data in enumerate(pre_dicts):

    outputs = results[idx]

    post_jsons = [
        process_output(text, data['doctext'])
        for text in outputs
    ]

    print(post_jsons)
    if n == 1:

        data['pred_json'] = post_jsons[0]

        if think:
            data['pred_reasoning'] = reasonings[idx][0]

    else:
        data['pred_json'] = post_jsons

        if think:
            data['pred_reasoning'] = reasonings[idx]


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------
os.makedirs(os.path.dirname(path_write), exist_ok=True)

with open(path_write, 'w') as output_file:
    for entry in pre_dicts:
        output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')


print(f'Final predictions written to: {path_write}')