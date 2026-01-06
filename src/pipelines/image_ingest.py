from dotenv import load_dotenv
load_dotenv()

from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image

import os
import pytesseract
import torch

from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain_core.documents import Document
from qdrant_client.models import PointStruct

from pipelines.ingest import make_chunk_id
from vectorstore.vectorstore import get_vectorstore
from embeddings.clip_embedder import CLIPEmbedder

IMAGE_DIR = Path(os.getenv("CLEAN_PATH")) / "images"
BLIP_MODEL = "Salesforce/blip-image-captioning-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
_blip_model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL).to(DEVICE)
_blip_model.eval()

_clip_embedder = CLIPEmbedder(device=DEVICE)

def parse_image_filename(image_path: Path) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    parts = image_path.stem.split("-")
    if len(parts) != 3:
        return None, None, None

    element_type, page, index = parts
    try:
        return element_type.capitalize(), int(page), int(index)
    except ValueError:
        return None, None, None

def ocr_image(image_path: Path) -> str:
    try:
        img = Image.open(image_path)
        return pytesseract.image_to_string(img).strip()
    except Exception:
        return ""

def generate_caption(image_path: Path) -> str:
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = _blip_processor(images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            output = _blip_model.generate(**inputs, max_new_tokens=50)
        return _blip_processor.decode(output[0], skip_special_tokens=True).strip()
    except Exception:
        return ""

def build_canonical_text(caption: str, ocr_text: str) -> str:
    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr_text:
        parts.append(f"OCR: {ocr_text}")
    return "\n".join(parts).strip()

def ingest_images(batch_size: int = 50):
    print("Running image ingestion and vectorstore upsert")
    vectorstore = get_vectorstore()
    image_docs: List[Document] = []

    if not IMAGE_DIR.exists():
        print("No images directory found")
        return image_docs

    points_batch = []

    for image_path in sorted(IMAGE_DIR.rglob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            continue

        element_type, page, index = parse_image_filename(image_path)
        if not element_type:
            continue

        ocr_text = ocr_image(image_path)
        caption = generate_caption(image_path)
        canonical_text = build_canonical_text(caption, ocr_text)

        metadata = {
            "source": str(image_path),
            "page": page,
            "element_type": element_type,
            "image_path": str(image_path),
            "index": index,
            "caption": caption,
            "ocr": ocr_text,
            "modality": "image",
        }

        doc = Document(
            page_content=canonical_text or f"[{element_type.upper()}]",
            metadata=metadata,
            id=make_chunk_id(str(image_path), page, canonical_text or f"{element_type}-{index}"),
        )

        try:
            img = Image.open(image_path).convert("RGB")
            image_vec = _clip_embedder.embed_image(img)
            text_vec = _clip_embedder.embed_text(canonical_text) if canonical_text else None
        except Exception as e:
            print(f"Embedding failed for {image_path}: {e}")
            continue

        vectors = {
            "image_dense": image_vec,
        }

        if text_vec is not None:
            vectors["image_text_dense"] = text_vec

        point = PointStruct(
            id=doc.id,
            vector=vectors,
            payload=metadata,
        )

        points_batch.append(point)
        image_docs.append(doc)

        if len(points_batch) >= batch_size:
            vectorstore.client.upsert(collection_name="embedded_data", points=points_batch)
            points_batch = []

    if points_batch:
        vectorstore.client.upsert(collection_name="embedded_data", points=points_batch)

    print(f"Ingested and upserted {len(image_docs)} image elements into vectorstore")
    return image_docs

if __name__ == "__main__":
    ingest_images()