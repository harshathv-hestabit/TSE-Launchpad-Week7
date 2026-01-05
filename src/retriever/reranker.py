from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from collections import defaultdict
from typing import List, Tuple


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        top_n_pages: int = 5,
    ):
        self.model = CrossEncoder(model_name)
        self.top_n_pages = top_n_pages

    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []

        # 1. Group chunks by (source, page)
        pages: dict[Tuple[str, int], List[Document]] = defaultdict(list)
        for d in docs:
            key = (
                d.metadata.get("source"),
                d.metadata.get("page"),
            )
            pages[key].append(d)

        # 2. Build page-level text
        page_docs: List[Tuple[Tuple[str, int], str, List[Document]]] = []
        for key, page_chunks in pages.items():
            page_text = "\n".join(c.page_content for c in page_chunks)
            page_docs.append((key, page_text, page_chunks))

        # 3. Cross-encode at page level
        pairs = [(query, page_text) for _, page_text, _ in page_docs]
        scores = self.model.predict(pairs)

        ranked_pages = sorted(
            zip(page_docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # 4. Return chunks from top-N pages (preserve chunk metadata)
        top_chunks: List[Document] = []
        for (key, _, page_chunks), _ in ranked_pages[: self.top_n_pages]:
            top_chunks.extend(page_chunks)

        return top_chunks