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
from utils import maxCommStr

#TODO:Better-ea bihurtu!
exit()

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Arguments required to the rejection sampling')
parser.add_argument('--split',            dest='split',            type=str)
parser.add_argument('--n',                dest='n',                type=int)
parser.add_argument('--language',         dest='language',         type=str)
parser.add_argument('--model-name',       dest='model_name',       type=str)
parser.add_argument('--out-dir',          dest='out_dir',          type=str)
parser.add_argument('--think',            dest='think',            action='store_true',
                    help='Enable reasoning via the Responses API. Requires a reasoning-capable '
                         'model (e.g. o3, o4-mini, gpt-5.5). Has no effect on gpt-4o.')
parser.add_argument('--reasoning-effort', dest='reasoning_effort', type=str, default='medium',
                    choices=['none', 'low', 'medium', 'high', 'xhigh'],
                    help='How hard the model thinks. Only used when --think is set.')
parser.add_argument('--api-key',          dest='api_key',          type=str, default=None)

parser.set_defaults(
    model_name="gpt-5.5",   # override with e.g. o4-mini or gpt-5.5 when using --think
    language="en",
    split="proba",
    n=1,
    think=False,
    reasoning_effort="medium",
)

args             = parser.parse_args()
split            = args.split
language         = args.language
n                = args.n
model_name       = args.model_name
think            = args.think
reasoning_effort = args.reasoning_effort
path_write       = args.out_dir

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------
api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No API key found. Pass --api-key or set OPENAI_API_KEY.")
client = OpenAI(api_key=api_key)

# ---------------------------------------------------------------------------
# Prompt factory
# NOTE: prompt_factory must expose generate_prompt_text(data) -> list[dict]
# returning an OpenAI-style messages list, e.g.:
#   [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
# The original generate_prompt() returned vLLM token IDs and cannot be used here.
# ---------------------------------------------------------------------------
LANGUAGE_MAP = {
    "en": "English", "ar": "Arabic", "fa": "Farsi",
    "ko": "Korean",  "ru": "Russian", "zh": "Chinese",
}
language_name = LANGUAGE_MAP[language]
prompt = prompt_factory(model_name, language_name, "MUC", think)

map_field = {
    "PerpInd": "A person responsible for the incident. (PerpInd)",
    "PerpOrg": "An organization responsible for the incident. (PerpOrg)",
    "Target":  "An inanimate object that was attacked. (Target)",
    "Victim":  "The name of a person who was the obvious or apparent target of the attack or who became a victim of the attack. (Victim)",
    "Weapon":  "A device used by the perpetrator(s) in carrying out the terrorist act. (Weapon)",
}

path_read = (
    "multimuc/data/multimuc_v1.0/corrected/"
    + language + "/" + split + "_simplified_preprocess.jsonl"
)
if os.path.exists(path_write):
    os.remove(path_write)

def enforce_no_additional_props(schema):
    if isinstance(schema, dict):

        if schema.get("type") == "object":
            schema["additionalProperties"] = False

        for value in schema.values():
            enforce_no_additional_props(value)

    elif isinstance(schema, list):
        for item in schema:
            enforce_no_additional_props(item)

    return schema

# ---------------------------------------------------------------------------
# JSON schema for structured outputs
# ---------------------------------------------------------------------------
json_schema = Base.model_json_schema() # type: ignore
json_schema = enforce_no_additional_props(json_schema)

# ---------------------------------------------------------------------------
# Helper: post-process a parsed JSON output (unchanged from original)
# ---------------------------------------------------------------------------
def process_output(json_text: str, doctext: str) -> list:
    post_templates = []
    lower_doc = doctext.lower()
    try:
        for template in json.loads(json_text)["templates"]:
            post_processed = {}
            for key in template.keys():
                if key != "incident_type" and template[key] != []:
                    post_processed[key] = []
                    for elem in template[key]:
                        lower_elem = elem.lower()
                        if lower_elem in lower_doc:
                            post_processed[key].append([lower_elem])
                        else:
                            commn_str = maxCommStr(lower_elem, lower_doc)
                            if commn_str:
                                if commn_str[0]  == " ": commn_str = commn_str[1:]
                                if commn_str[-1] == " ": commn_str = commn_str[:-1]
                                post_processed[key].append([commn_str])
                else:
                    post_processed[key] = template[key]
            post_templates.append(post_processed)
    except Exception:
        post_templates.append("ERROR")
    return post_templates


# ---------------------------------------------------------------------------
# --think path: Responses API
#
#   The Responses API is the only way to get all three in a single call:
#     1. Reasoning:        model thinks internally before answering
#     2. Reasoning summary: a human-readable summary of the thinking trace,
#                          returned as a separate output item
#     3. Structured output: JSON constrained by the schema via text.format
#
#   n > 1 is not supported for reasoning models, so we loop.
#   temperature and top_p must NOT be set for reasoning models.
# ---------------------------------------------------------------------------
def run_think_call(messages: list) -> tuple[str | None, str]:
    """Returns (reasoning_summary, output_text)."""
    response = client.responses.create(
        model=model_name,
        input=messages,
        reasoning={"effort": reasoning_effort, "summary": "detailed"},
        text={"format": {
            "type":   "json_schema",
            "name":   "MUC",
            "schema": json_schema,
            "strict": True,
        }},
        prompt_cache_retention="24h",
        max_output_tokens=7000,
        #seed=42,
        #temperature=0.0,
    )

    reasoning_summary = None
    output_text = ""
    for item in response.output:
        if item.type == "reasoning" and item.summary:
            reasoning_summary = " ".join(
                block.text for block in item.summary if hasattr(block, "text")
            )
        elif item.type == "message":
            for block in item.content:
                if hasattr(block, "text"):
                    output_text += block.text

    return reasoning_summary, output_text


# ---------------------------------------------------------------------------
# no-think path: Chat Completions API
#
#   Native n > 1 support in a single call.
#   temperature and top_p can be added here if needed.
# ---------------------------------------------------------------------------
def run_no_think_call(messages: list) -> list[str]:
    """Returns a list of n output texts."""
    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        n=n,
        max_completion_tokens=1200,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name":   "extraction_output",
                "schema": json_schema,
                "strict": True,
            },
        },
    )
    return [choice.message.content for choice in completion.choices]


# ---------------------------------------------------------------------------
# Main inference dispatcher
# ---------------------------------------------------------------------------
def run_inference(messages: list, doctext: str) -> tuple[list, list]:
    """Returns (reasonings, post_processed_jsons), each a list of length n."""
    if think:
        reasonings, post_jsons = [], []
        for _ in range(n):
            summary, text = run_think_call(messages)
            reasonings.append(summary)
            post_jsons.append(process_output(text, doctext))
    else:
        outputs    = run_no_think_call(messages)
        reasonings = [None] * n
        post_jsons = [process_output(text, doctext) for text in outputs]
    
    if n==1:
        reasonings = reasonings[0]
        post_jsons = post_jsons[0]

    return reasonings, post_jsons


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
pre_dicts = []

with open(path_read, 'r') as file:
    for line in file:
        pre_dicts.append(json.loads(line))

for idx, data in tqdm(enumerate(pre_dicts)):
    # generate_prompt_text must return an OpenAI messages list (see note above)
    messages = prompt.generate_prompt(data)

    reasonings, post_jsons = run_inference(messages, data["doctext"])

    pre_dicts[idx]["pred_reasoning"] = reasonings
    pre_dicts[idx]["pred_json"]      = post_jsons


print("Done")
with open(path_write, 'w') as output_file:
    for entry in pre_dicts:
        output_file.write(json.dumps(entry, ensure_ascii=False) + '\n')