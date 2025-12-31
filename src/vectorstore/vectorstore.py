from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from langchain_qdrant import QdrantVectorStore
from src.embeddings.embedder import get_embeddings

_client = None
_vectorstore = None

COLLECTION_NAME = "embedded_data"
QDRANT_PATH = "src/vectorstore/qdrant"

def get_vectorstore():
    global _client, _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()
    _client = QdrantClient(path=QDRANT_PATH)

    existing_collections = {
        c.name for c in _client.get_collections().collections
    }

    if COLLECTION_NAME not in existing_collections:
        vector_size = len(embeddings.embed_query("vector_size_probe"))

        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

    _vectorstore = QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )

    return _vectorstore