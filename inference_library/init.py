from prompts import *
PROMPT_FN = {k: globals()[k] for k in globals()}