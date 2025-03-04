import requests
import json 
template = {"templates": [{"incident_type": "bombing", "PerpInd": [], "PerpOrg": ["proba"], "Target": [], "Victim": [], "Weapon": ["car bomb"]}, {"incident_type": "bombing", "PerpInd": [], "PerpOrg": ["proba"], "Target": [], "Victim": [], "Weapon": ["car bomb"]}]}
completions = [json.dumps(template) for i in range(1200)]
template_truth = [{"incident_type": "bombing", "PerpInd": [], "PerpOrg": [], "Target": [], "Victim": [], "Weapon": [[["car bomb", 2812]]]}]
ground_truths = [json.dumps(template_truth) for i in range(1200)]
response = requests.post('http://localhost:4416/reward', 
    json={'completions': completions, 'ground_truths': ground_truths, "reasoning": False})
print(response.json())