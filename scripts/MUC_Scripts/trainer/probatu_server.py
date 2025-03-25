import requests
import json 
template = {"templates": []}
completions = [json.dumps(template) for i in range(2)]
template_truth = []
ground_truths = [json.dumps(template_truth) for i in range(2)]
response = requests.post('http://localhost:4416/reward', 
    json={'completions': completions, 'ground_truths': ground_truths, "reasoning": False})
print(response.json())