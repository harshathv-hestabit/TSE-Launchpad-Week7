import json
import time
import uuid
import yaml

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from pathlib import Path
from PIL import Image
from typing import Optional

from generator.llm_client import LLM, LLMCallable
from retriever.hybrid_retriever import hybrid_retrieve
from memory.memory_store import MemoryStore
from evaluation.rag_eval import (hallucination_score,refine_answer,confidence_score)
from retriever.image_search import ImageSearcher
from pipelines.sql_pipeline import SQLQAPipeline

with open(Path("src/config/model.yaml")) as f:
    config = yaml.safe_load(f)

LOG_FILE = Path("src/logs/CHAT-LOGS.json")
MEM_FILE = Path("src/logs/memory.json")

llm_client = LLM(config).get_model()
llm = LLMCallable(llm_client, model_name=config["model_name"])
memory = MemoryStore(k=5)
image_searcher = ImageSearcher()
sql_pipeline = SQLQAPipeline(db_path="financial_reports.db",llm=llm)

app = FastAPI()

class AskRequest(BaseModel):
    question: str

def log_event(payload: dict):
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(payload) + "\n")

def log_memory(data):
    with MEM_FILE.open("a") as f:
        f.write(json.dumps(data) + "\n")

@app.post("/ask")
def ask(req: AskRequest):
    start = time.time()
    request_id = str(uuid.uuid4())

    memory.add_user(req.question)

    docs = hybrid_retrieve(req.question, top_k=5)
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
Use the context below to answer.

Context:
{context}

Conversation history:
{memory.get()}

Question:
{req.question}
"""

    draft = llm(prompt)
    refined = refine_answer(llm, req.question, draft, docs)

    hallucination = hallucination_score(refined, docs)
    confidence = confidence_score(llm, req.question, refined, docs)

    memory.add_assistant(refined)

    log_event({
        "id": request_id,
        "endpoint": "/ask",
        "question": req.question,
        "answer": refined,
        "context":context,
        "hallucination": hallucination,
        "confidence": confidence,
        "latency_ms": int((time.time() - start) * 1000),
    })

    return {
        "answer": refined,
        "confidence": confidence,
        "hallucination_score": hallucination,
    }

def build_context(results):
    context_lines = []
    for r in results:
        line_parts = []

        if r.get("page_content"):
            line_parts.append(r["page_content"])
        if r.get("caption"):
            line_parts.append(f"Caption: {r['caption']}")
        if r.get("ocr"):
            line_parts.append(f"OCR: {r['ocr']}")
        if r.get("source"):
            line_parts.append(f"Source: {r['source']}")
        if line_parts:
            context_lines.append(" | ".join(line_parts))

    return "\n".join(context_lines)

@app.post("/ask-image")
async def ask_image(
    question: str = Form(...),
    mode: str = Form("image_to_text"),
    file: Optional[UploadFile] = File(None)
):
    start = time.time()
    request_id = str(uuid.uuid4())

    if file:
        img = Image.open(file.file).convert("RGB")
        if mode == "image":
            results = image_searcher.search_by_image(img)
        elif mode == "image_to_text":
            results = image_searcher.search_image_to_text(img)
        else:
            results = image_searcher.search_image_to_text(img)
    else:
        results = image_searcher.search_by_text(question)

    context = build_context(results)
    if not context.strip():
        draft = "No usable information was found in the retrieved images to answer this question."
    else:
        draft = llm(f"""
You are an assistant answering questions about images using retrieval results.

Rules:
- Use captions, OCR text, and metadata as valid evidence but dont mention caption and ocr specifically. use only the data values.
- If visual details are missing, infer purpose or meaning from available text.
- Do NOT mention missing context, uncertainty, or limitations unless the context is completely empty.
- Do NOT describe what you cannot see.
- Write a natural, user-friendly answer in plain language.
- Be concise and factual.

Retrieved image information:
{context}

User question:
{question}
""")

    hallucination = 0.0 if context else 1.0
    confidence = 0.9 if context else 0.3

    memory.add_user(question)
    memory.add_assistant(draft)

    log_event({
        "id": request_id,
        "endpoint": "/ask-image",
        "question": question,
        "answer": draft,
        "context":context,
        "confidence": confidence,
        "hallucination": hallucination,
        "latency_ms": int((time.time() - start) * 1000),
    })

    return {
        "answer": draft,
        "confidence": confidence,
        "matches": results,
    }

@app.post("/ask-sql")
def ask_sql(req: AskRequest):
    start = time.time()
    request_id = str(uuid.uuid4())

    result = sql_pipeline.run(req.question)

    log_event({
        "id": request_id,
        "endpoint": "/ask-sql",
        "question": req.question,
        "answer": result["answer"],
        "confidence": result["confidence"],
        "context":result["context"],
        "latency_ms": int((time.time() - start) * 1000),
    })

    return result