from typing import List, Union
from pathlib import Path
from PIL import Image

from embeddings.clip_embedder import CLIPEmbedder
from qdrant_client import QdrantClient

class ImageSearcher:
    def __init__(self):
        self.client = QdrantClient(path="src/vectorstore/qdrant")
        self.collection_name = "embedded_data"
        self.clip = CLIPEmbedder()

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image
        elif isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

    def search_by_text(self, query: str, limit: int = 5) -> List[dict]:
        if not query or not query.strip():
            return []

        try:
            query_vec = self.clip.embed_text(query)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                using="image_text_dense",
                limit=limit,
                with_payload=True,
            )

            return self._format_results(response)
        except Exception as e:
            print(f"Error in search_by_text: {e}")
            return []

    def search_by_image(self,image: Union[str, Path, Image.Image],limit: int = 15) -> List[dict]:
        try:
            img = self._load_image(image)
            query_vec = self.clip.embed_image(img)

            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                using="image_dense",
                limit=limit,
                with_payload=True,
            )

            return self._format_results(response)
        except Exception as e:
            print(f"Error in search_by_image: {e}")
            return []

    def search_image_to_text(self,image: Union[str, Path, Image.Image],limit: int = 15) -> List[dict]:
        try:
            img = self._load_image(image)
            query_vec = self.clip.embed_image(img)
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                using="image_text_dense",
                limit=limit,
                with_payload=True,
            )
            return self._format_results(response)
        except Exception as e:
            print(f"Error in search_image_to_text: {e}")
            return []

    def _format_results(self, response) -> List[dict]:
        formatted = []

        if not hasattr(response, 'points') or not response.points:
            return formatted
        
        print(response)
        for point in response.points:
            payload = point.payload or {}
            
            if isinstance(payload.get('metadata'), dict):
                metadata = payload.get('metadata', {})
                page_content = payload.get('page_content', '')
            else:
                metadata = payload
                page_content = ''
            
            score = point.score if hasattr(point, 'score') else 0.0            
            formatted.append(
                {
                    "id": point.id,
                    "score": round(float(score), 4),
                    "image_path": metadata.get("image_path"),
                    "source": metadata.get("source"),
                    "page": metadata.get("page"),
                    "element_type": metadata.get("element_type"),
                    "has_caption": metadata.get("has_caption"),
                    "has_ocr": metadata.get("has_ocr"),
                    "page_content": page_content,
                }
            )
        return formatted