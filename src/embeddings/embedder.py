import os
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL"),
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
