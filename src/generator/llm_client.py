from dotenv import load_dotenv
load_dotenv()

import os

from transformers import AutoModelForCausalLM
from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

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
