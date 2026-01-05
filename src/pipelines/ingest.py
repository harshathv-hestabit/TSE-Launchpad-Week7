from dotenv import load_dotenv
load_dotenv()

import os
import json
import uuid
from pathlib import Path
from typing import List

import yaml
from unstructured.partition.pdf import partition_pdf
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer

from generator.llm_client import LLM
from vectorstore.vectorstore import get_vectorstore

with open(Path("src/config/model.yaml"), "r") as f:
    config = yaml.safe_load(f)

DATA_PATH = Path(os.getenv("DATA_PATH"))
CLEANED_PATH = Path(os.getenv("CLEAN_PATH"))
CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH"))

IMAGE_DIR = CLEANED_PATH/"images"
CLEANED_FILE = CLEANED_PATH / "pages.jsonl"
CHUNKS_FILE = CHUNKS_PATH / "chunks.jsonl"

NAMESPACE_CHUNKS = uuid.UUID("12345678-1234-5678-1234-567812345678")

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        mode="fast"
    )
    return tokenizer

def make_chunk_id(source: str, page: int, text: str) -> str:
    raw = f"{source}|{page}|{text}"
    return str(uuid.uuid5(NAMESPACE_CHUNKS, raw))

def normalize_text(text: str) -> str:
    return text.replace("\r\n", "\n").strip()

def chunk_text(text: str, text_splitter: TokenTextSplitter, strategy: str = "recursive") -> List[str]:
    chunks: List[str] = []

    if strategy == "recursive":
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            sub_texts = text_splitter.split_text(para) if len(para.split()) > 250 else [para]
            chunks.extend(sub_texts)

    elif strategy == "layout-aware":
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        buffer: List[str] = []
        for line in lines:
            if len(line.split()) < 5 and (line.isupper() or line.endswith(":")):
                if buffer:
                    sub_texts = text_splitter.split_text("\n".join(buffer))
                    chunks.extend(sub_texts)
                    buffer = []
                chunks.append(line) 
            else:
                buffer.append(line)
        if buffer:
            sub_texts = text_splitter.split_text("\n".join(buffer))
            chunks.extend(sub_texts)
    else:
        chunks = [text]

    return chunks

def process_pdf(file_path: Path, text_splitter: TokenTextSplitter, strategy: str = "recursive") -> List[Document]:
    elements = partition_pdf(
        filename=file_path,
        languages=["eng"],
        include_page_breaks=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=IMAGE_DIR
    )

    chunks: List[Document] = []
    for element in elements:
        el_text = getattr(element, "text", None)
        if not el_text:
            continue

        el_text = normalize_text(el_text)
        sub_texts = chunk_text(el_text, text_splitter, strategy=strategy)

        for chunk in sub_texts:
            metadata = {
                "source": str(file_path),
                "page": getattr(element.metadata, "page_number", None),
                "element_type": getattr(element, "category", "Unknown"),
                "coordinates": getattr(element.metadata, "coordinates", None)
            }
            chunks.append(
                Document(
                    page_content=chunk,
                    metadata=metadata,
                    id=make_chunk_id(str(file_path), metadata["page"], chunk)
                )
            )
    return chunks

def ingest(chunking_strategy: str = "recursive"):
    if CLEANED_FILE.exists():
        print("Loading cleaned documents")
        documents: List[Document] = []
        with open(CLEANED_FILE, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                documents.append(
                    Document(
                        page_content=obj["content"],
                        metadata=obj["metadata"]
                    )
                )
    else:
        print("Cleaning data with unstructured")
        documents: List[Document] = []
        for pdf_file in sorted(DATA_PATH.rglob("*.pdf")):
            print(f"Processing {pdf_file}")
            text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
                tokenizer=get_tokenizer(),
                chunk_size=690,
                chunk_overlap=0
            )
            pdf_chunks = process_pdf(pdf_file, text_splitter, strategy=chunking_strategy)
            documents.extend(pdf_chunks)

        CLEANED_PATH.mkdir(parents=True, exist_ok=True)
        with open(CLEANED_FILE, "w", encoding="utf-8") as f:
            for doc in documents:
                metadata = doc.metadata.copy()
                metadata.pop("coordinates", None) 
                f.write(
                    json.dumps(
                        {"content": doc.page_content, "metadata": metadata},
                        ensure_ascii=False
                    ) + "\n"
                )

    print(f"Total documents after cleaning: {len(documents)}")

    if CHUNKS_FILE.exists():
        print("Loading processed chunks from cache")
        chunks: List[Document] = []
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                chunks.append(
                    Document(
                        page_content=obj["content"],
                        metadata=obj["metadata"],
                        id=obj.get("id")
                    )
                )
    else:
        print("Creating layout-aware chunks and saving to file")
        CHUNKS_PATH.mkdir(parents=True, exist_ok=True)
        chunks: List[Document] = []
        text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
            tokenizer=get_tokenizer(),
            chunk_size=690,
            chunk_overlap=0
        )

        for doc in documents:
            sub_texts = chunk_text(doc.page_content, text_splitter, strategy=chunking_strategy)
            for sub_text in sub_texts:
                metadata = doc.metadata.copy()
                metadata.pop("coordinates", None)
                chunks.append(
                    Document(
                        page_content=sub_text,
                        metadata=metadata,
                        id=make_chunk_id(metadata["source"], metadata["page"], sub_text)
                    )
                )

        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(
                        {"content": chunk.page_content, "metadata": chunk.metadata, "id": chunk.id},
                        ensure_ascii=False
                    ) + "\n"
                )

    print(f"Total number of chunks: {len(chunks)}")

    print("Adding chunks to vectorstore")
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    collection = vectorstore.client.get_collection("embedded_data")
    print(f"Vectorstore points count: {collection.points_count}")
    print(f"Successfully added {len(chunks)} chunks to vectorstore")

if __name__ == "__main__":
    print("Running Ingestion Pipeline")
    ingest(chunking_strategy="layout-aware")