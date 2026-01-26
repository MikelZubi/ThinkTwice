class Hyperparameters():
    def __init__(self, model_name, think, n):
        #VLLM's Default values:
        self.temperature=1.0
        self.top_p=1.0
        self.top_k=-1
        self.min_p=0.0
        if n > 1:
            if "R1" in model_name:
                self.temperature=0.6
            elif "Qwen3" in model_name:
                if think:
                    self.temperature=0.6 
                    self.top_p=0.95
                    self.top_k=20
                    self.min_p=0
                else:
                    self.temperature=0.7
                    self.top_p=0.8
                    self.top_k=20
                    self.min_p=0
            else: #Standart Llama3.3
                self.temperature=0.7
        else:
            self.temperature=0.0