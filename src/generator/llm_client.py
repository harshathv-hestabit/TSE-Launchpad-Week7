from transformers import AutoModelForCausalLM, AutoTokenizer

class LLM:
    def __init__(self,config):
        self.config = config
        self.provider = config["provider"]
        self.model_name = config["model_name"]
        self.api_key_env = config["api_key_env"]
        self._model = None
        self._tokenizer = None
    def get_model(self):
        if self.provider == "local":    
            if self._model is None:
                self._model = AutoModelForCausalLM.from_pretrained(self.model_name,device_map="auto",dtype="auto")
            return self._model
    def get_tokenizer(self):
        if self.provider == "local":
            if self._tokenizer is None:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name,use_fast=True)
            return self._tokenizer