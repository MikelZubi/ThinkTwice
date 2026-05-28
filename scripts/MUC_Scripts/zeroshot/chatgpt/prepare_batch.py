from openai import OpenAI
import json
import os
import argparse
from tqdm import tqdm

import sys
sys.path.append("class_data")
sys.path.append("inference_library")

from MUC_Class_simplified import *
from prompt_factory import prompt_factory


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Prepare OpenAI batch requests')

parser.add_argument('--split',            type=str)
parser.add_argument('--n',                type=int, default=1)
parser.add_argument('--language',         type=str)
parser.add_argument('--model-name',       type=str)
parser.add_argument('--batch-file',       type=str, default='batch_requests.jsonl')
parser.add_argument('--think',            action='store_true')
parser.add_argument('--reasoning-effort', type=str, default='medium')

parser.set_defaults(
    model_name='gpt-5.5',
    language='en',
    split='proba',
)

args = parser.parse_args()

split            = args.split
language         = args.language
n                = args.n
model_name       = args.model_name
think            = args.think
reasoning_effort = args.reasoning_effort
batch_file       = args.batch_file


# ---------------------------------------------------------------------------
# Prompt factory
# ---------------------------------------------------------------------------
LANGUAGE_MAP = {
    'en': 'English',
    'ar': 'Arabic',
    'fa': 'Farsi',
    'ko': 'Korean',
    'ru': 'Russian',
    'zh': 'Chinese',
}

language_name = LANGUAGE_MAP[language]
prompt = prompt_factory(model_name, language_name, 'MUC', think)


# ---------------------------------------------------------------------------
# Input path
# ---------------------------------------------------------------------------
path_read = (
    'multimuc/data/multimuc_v1.0/corrected/'
    + language + '/' + split + '_simplified_preprocess.jsonl'
)


# ---------------------------------------------------------------------------
# JSON schema helper
# ---------------------------------------------------------------------------
def enforce_no_additional_props(schema):
    if isinstance(schema, dict):

        if schema.get('type') == 'object':
            schema['additionalProperties'] = False

        for value in schema.values():
            enforce_no_additional_props(value)

    elif isinstance(schema, list):
        for item in schema:
            enforce_no_additional_props(item)

    return schema


json_schema = Base.model_json_schema()
json_schema = enforce_no_additional_props(json_schema)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
pre_dicts = []

with open(path_read, 'r') as file:
    for line in file:
        pre_dicts.append(json.loads(line))


# ---------------------------------------------------------------------------
# Create batch file
# ---------------------------------------------------------------------------
with open(batch_file, 'w') as fout:

    for idx, data in tqdm(enumerate(pre_dicts), total=len(pre_dicts)):

        messages = prompt.generate_prompt(data)

        for sample_idx in range(n):

            custom_id = f'doc_{idx}_sample_{sample_idx}'

            if think:
                request = {
                    'custom_id': custom_id,
                    'method': 'POST',
                    'url': '/v1/responses',
                    'body': {
                        'model': model_name,
                        'input': messages,
                        'reasoning': {
                            'effort': reasoning_effort,
                            'summary': 'detailed',
                        },
                        'text': {
                            'format': {
                                'type': 'json_schema',
                                'name': 'MUC',
                                'schema': json_schema,
                                'strict': True,
                            }
                        },
                        'max_output_tokens': 7000,
                    }
                }

            else:
                request = {
                    'custom_id': custom_id,
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {
                        'model': model_name,
                        'messages': messages,
                        'max_completion_tokens': 1200,
                        'response_format': {
                            'type': 'json_schema',
                            'json_schema': {
                                'name': 'extraction_output',
                                'schema': json_schema,
                                'strict': True,
                            },
                        },
                    }
                }

            fout.write(json.dumps(request) + '\n')


print(f'Batch file written to: {batch_file}')
