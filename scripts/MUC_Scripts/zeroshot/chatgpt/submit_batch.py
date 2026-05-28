from openai import OpenAI
import argparse
import os


parser = argparse.ArgumentParser(description='Submit OpenAI batch job')

parser.add_argument('--batch-file', type=str, required=True)
parser.add_argument('--think', action='store_true')
parser.add_argument('--api-key', type=str, default=None)

args = parser.parse_args()

api_key = args.api_key or os.environ.get('OPENAI_API_KEY')

if not api_key:
    raise ValueError('OPENAI_API_KEY not found')


client = OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Upload file
# ---------------------------------------------------------------------------
print('Uploading batch file...')

batch_input_file = client.files.create(
    file=open(args.batch_file, 'rb'),
    purpose='batch'
)

print(f'Uploaded file ID: {batch_input_file.id}')


# ---------------------------------------------------------------------------
# Select endpoint
# ---------------------------------------------------------------------------
endpoint = '/v1/responses' if args.think else '/v1/chat/completions'


# ---------------------------------------------------------------------------
# Create batch
# ---------------------------------------------------------------------------
print('Creating batch job...')

batch_job = client.batches.create(
    input_file_id=batch_input_file.id,
    endpoint=endpoint,
    completion_window='24h'
)


print('\n=== BATCH CREATED ===')
print(f'Batch ID: {batch_job.id}')
print(f'Status:   {batch_job.status}')

id_file_path = f"{args.batch_file.replace('.jsonl', '')}_id.txt"
with open(id_file_path, 'w') as f:
    f.write(batch_job.id)
print(f'Batch ID saved to: {id_file_path}')
