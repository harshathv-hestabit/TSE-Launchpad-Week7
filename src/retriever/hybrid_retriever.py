from typing import Optional, List, Dict, Any
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document
from vectorstore.vectorstore import get_vectorstore
from retriever.reranker import CrossEncoderReranker

def build_qdrant_filter(filters: Optional[Dict[str, Any]] = None) -> Optional[Filter]:
    if not filters:
        return None

    conditions = [FieldCondition(key=key, match=MatchValue(value=value)) for key, value in filters.items()]
    return Filter(must=conditions)

def get_hybrid_retriever(top_k: int = 5,filters: Optional[Dict[str, Any]] = None):
    vectorstore = get_vectorstore()
    qdrant_filter = build_qdrant_filter(filters)

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": top_k,
            "filter": qdrant_filter,
        },
    )

def hybrid_retrieve(query: str,top_k: int = 5,filters: Optional[Dict[str, Any]] = None) -> List[Document]:
    reranker = CrossEncoderReranker()
    retriever = get_hybrid_retriever(top_k=top_k, filters=filters)
    results = retriever.invoke(query)

    if not results:
        fallback_retriever = get_hybrid_retriever(top_k=top_k, filters=None)
        results = fallback_retriever.invoke(query)
    
    results = reranker.rerank(docs=results,query=query)
    return results