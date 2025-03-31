from prompts import *
PROMPT_FN = {k: globals()[k] for k in globals() if k.startswith("P_") or k.startswith("Response")}