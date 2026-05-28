from openai import OpenAI
import argparse
import os


parser = argparse.ArgumentParser(description='Check OpenAI batch status')

parser.add_argument('--batch-id-file', type=str, required=True)
parser.add_argument('--api-key', type=str, default=None)

args = parser.parse_args()

api_key = args.api_key or os.environ.get('OPENAI_API_KEY')

if not api_key:
    raise ValueError('OPENAI_API_KEY not found')


client = OpenAI(api_key=api_key)

with open(args.batch_id_file, 'r') as f:
    batch_id = f.read().strip()

batch = client.batches.retrieve(batch_id)

print('\n=== BATCH STATUS ===')
print(f'ID:              {batch.id}')
print(f'Status:          {batch.status}')
print(f'Created at:      {batch.created_at}')
print(f'Completed at:    {batch.completed_at}')
print(f'Failed at:       {batch.failed_at}')
print(f'Request counts:  {batch.request_counts}')

if batch.output_file_id:
    print(f'Output file ID:  {batch.output_file_id}')

if batch.error_file_id:
    print(f'Error file ID:   {batch.error_file_id}')