from typing import List, Tuple
from langchain_core.documents import Document
from collections import defaultdict
import re


class ContextBuilder:
    def __init__(
        self,
        max_pages: int = 5,
        max_chars_per_page: int = 1500,
        separator: str = "\n\n---\n\n",
    ):
        self.max_pages = max_pages
        self.max_chars_per_page = max_chars_per_page
        self.separator = separator

    def _remove_noise(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
        text = re.sub(r"\|\s+Downer EDI Limited", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _format_tables(self, text: str) -> str:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        formatted = []
        buffer = []

        for line in lines:
            if re.fullmatch(r"[-–\d,.\s$’m]+", line):
                buffer.append(line)
            else:
                if buffer:
                    formatted.append(" | ".join(buffer))
                    buffer = []
                formatted.append(line)

        if buffer:
            formatted.append(" | ".join(buffer))

        return "\n".join(formatted)

    def _smart_truncate(self, text: str) -> str:
        if len(text) <= self.max_chars_per_page:
            return text

        cut = text.rfind(".", 0, self.max_chars_per_page)
        return text[: cut + 1] if cut != -1 else text[: self.max_chars_per_page]

    def build(self, documents: List[Document]) -> str:
        # 1. Group by page
        pages: dict[Tuple[str, int], List[str]] = defaultdict(list)

        for doc in documents:
            key = (
                doc.metadata.get("source", "unknown"),
                doc.metadata.get("page", "unknown"),
            )
            pages[key].append(doc.page_content)

        blocks = []

        # 2. Build page-level context
        for (source, page), texts in list(pages.items())[: self.max_pages]:
            page_text = "\n".join(texts)
            page_text = self._remove_noise(page_text)
            page_text = self._format_tables(page_text)
            page_text = self._smart_truncate(page_text)

            block = f"[SOURCE: {source} | PAGE: {page}]\n{page_text}"
            blocks.append(block)

        return self.separator.join(blocks)