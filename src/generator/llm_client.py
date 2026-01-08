from dotenv import load_dotenv
load_dotenv()

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError
from deepeval.models.base_model import DeepEvalBaseLLM

class LLM:
    def __init__(self,config):
        self.config = config
        self.provider = config["provider"]
        self.model_name = config["model_name"]
        self.api_key = os.getenv("HF_TOKEN")
        self._model = None

    def get_model(self):
        if self._model is not None:
            return self._model

        if self.provider == "api":
            try:
                self._model = self._api_client()
                return self._model
            except (HfHubHTTPError, RuntimeError, OSError) as e:
                print("HF inference failed, falling back to local model:", str(e))
                self._model = self._local_client()
                return self._model

        self._model = self._local_client()
        return self._model

    def _api_client(self):
        if not self.api_key:
            raise RuntimeError("HF access token not found")

        return InferenceClient(
            model=self.model_name,
            token=self.api_key,
        )

    def _local_client(self):
        print("Loading local LLM")
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype="auto",
        )
class LLMCallable:
    def __init__(self, llm_client, model_name):
        self.client = llm_client
        self.model_name = model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except Exception:
            self.tokenizer = None

    def __call__(self, prompt: str) -> str:
        if hasattr(self.client, "chat"):
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=256,
                temperature=0,
            )
            return response.choices[0].message.content.strip()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.client.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0,
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
class HFEvalModel(DeepEvalBaseLLM):
    def __init__(self, llm_callable, model_name: str):
        self.llm = llm_callable
        self.model_name = model_name

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        return self.llm(prompt)

    async def a_generate(self, prompt: str) -> str:
        return self.llm(prompt)

    def get_model_name(self):
        return self.model_name