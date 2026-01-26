with open("Docs/MUC_simplified.md", 'r') as md_file:
    MUC_GUIDELINES = md_file.read()

with open("Docs/BETTER_simplified.md", 'r') as md_file:
    BETTER_GUIDELINES = md_file.read()



#Prompts
P_U_70BR1_STEPS_REASONING = 'You are an expert in information extraction, you need to extract the information of the document that is provided in {language} as a template in JSON format. The guidelines for the dataset you need to extract are the followings:\n\n {guidelines} \n\nTo create the templates you need to follow two steps: first, you need to identify the incidents. Then, you need to fill the fields for each incident (or leave them empty). Create the template for the next document:\n{document}'
P_U_70BR1_REASONING = 'You are an expert in information extraction, you need to extract the information of the document that is provided in {language} as a template in JSON format. The guidelines for the dataset you need to extract are the followings:\n\n {guidelines} \n\nCreate the template for the next document:\n{document}'
P_S_LLAMA_JSON = 'You are an expert in information extraction, you need to extract the information of the document that is provided in {language} as a template in JSON format. The guidelines for the dataset you need to extract are the followings:\n\n {guidelines}'
P_U_LLAMA_JSON = 'Create the template for the next document:\n{document}'
P_S_LLAMA_JSON_REWARD = 'You are an expert in information extraction, you need to extract the information of the document that is provided in {language} as a template in JSON format.'
P_U_LLAMA_JSON_REWARD = 'Create the template for the next document:\n{document}'
#P_S_MUC_LLAMA_SCORER = 'You are an expert in information extraction, we have some sampled templates and your task is to select which one is the most accurated taking into account the guidelines and the document given in {language}. The guidelines for the dataset you need to extract are the followings:\n\n {guidelines}
U_MUC_LLAMA_SCORER = 'The document is the next one:\n{document} \n\n And the possible templates are the followings (the number at the start of each template indicates how many times was that template sampled):\n{templates} \n\n Select the most accurated one.'
#P_U_MUC_70BR1_SCORER = 'You are an expert in information extraction, we have some sampled templates and your task is to select which one is the most accurated taking into account the guidelines and the document given in {language}. The guidelines for the dataset you need to extract are the followings:\n\n {guidelines} \n\nThe document is the next one:\n{document} \n\n And the possible templates are the followings (the number at the start of each template indicates how many times was that template sampled):\n{templates} \n\n Select the most accurated one.'
P_U_8BR1_REASONING = '<USER> You are an expert in information extraction, you need to extract the information of the document that is provided in {language} as a template in JSON format. The guidelines for the dataset you need to extract are the followings:\n\n {guidelines} \n\nCreate the template for the next document:\n{document} </USER>\n'
P_A_8BR1_REASONING = '<ASSISTANT> {reasoning} </ASSISTANT>'
P_U_70BR1_OBTAIN_REASONING = 'You are an expert in information extraction, you have already extracted the information of the document that is provided in {language} as a template in JSON format, now you need to create the reasoning that explains this extraction. The guidelines for the dataset you have extracted are the followings:\n\n {guidelines} \n\n The document is the following:\n{document} \n\n The template is the following:\n{template} \n\n Create clear reasoning that explains how the template was derived from the document according to the guidelines.'
P_S_QWEN_JSON = P_S_LLAMA_JSON
P_U_QWEN_JSON = P_U_LLAMA_JSON