from typing import Union, List
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class CLIPEmbedder:
    def __init__(self,model_name: str = "openai/clip-vit-base-patch32",device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.embedding_dim = self.model.config.projection_dim

    def embed_image(self,image: Union[str, Path, Image.Image]) -> List[float]:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        inputs = self.processor(images=image,return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)

        features = self._normalize(features)
        return features[0].cpu().tolist()

    def embed_text(self, text: str) -> List[float]:
        inputs = self.processor(text=text,return_tensors="pt",padding=True,truncation=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

        features = self._normalize(features)
        return features[0].cpu().tolist()

    def _normalize(self,tensor: torch.Tensor) -> torch.Tensor:
        return tensor / tensor.norm(dim=-1, keepdim=True)