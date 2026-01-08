import re
from typing import List
from langchain_core.documents import Document

def context_coverage(answer: str, docs: List[Document]) -> float:
    if not docs:
        return 0.0

    context_text = " ".join(d.page_content for d in docs).lower()
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))

    if not answer_tokens:
        return 0.0

    hits = sum(1 for t in answer_tokens if t in context_text)
    return hits / len(answer_tokens)


def hallucination_score(answer: str, docs: List[Document]) -> float:
    coverage = context_coverage(answer, docs)
    return round(1.0 - coverage, 3)


def refine_answer(llm, question: str, draft: str, docs: List[Document]) -> str:
    context = "\n\n".join(d.page_content for d in docs)

    prompt = f"""
You are reviewing an answer for factual grounding.

Question:
{question}

Retrieved context:
{context}

Draft answer:
{draft}

If the draft contains unsupported claims, correct them.
Otherwise, improve clarity without adding new facts.
Return ONLY the revised answer.
"""

    return llm(prompt).strip()


def confidence_score(llm, question: str, answer: str, docs: List[Document]) -> float:
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
Evaluate confidence from 0 to 1.

Question:
{question}

Answer:
{answer}

Context:
{context}

Score based on grounding, clarity, and completeness.
Return ONLY a number between 0 and 1.
"""

    raw = llm(prompt).strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except Exception:
        return 0.5