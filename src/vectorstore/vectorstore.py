from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams
from langchain_qdrant import QdrantVectorStore, RetrievalMode, fastembed_sparse
from src.embeddings.embedder import get_embeddings
from embeddings.clip_embedder import CLIPEmbedder
from PIL import Image

_client = None
_vectorstore = None

COLLECTION_NAME = "embedded_data"
QDRANT_PATH = "src/vectorstore/qdrant"
SPARSE_MODEL = "qdrant/bm25"

def get_vectorstore():
    global _client, _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()
    sparse_embeddings = fastembed_sparse.FastEmbedSparse(model_name=SPARSE_MODEL)
    clip_embedder = CLIPEmbedder()
    
    # _client = QdrantClient(path=QDRANT_PATH) # Previously this was used for Local Qdrant Usuage
    _client = QdrantClient(url="http://localhost:6333")  #Used a qdrant docker image to run on this port, to support concurrent request from fastapi backend

    existing_collections = {
        c.name for c in _client.get_collections().collections
    }

    if COLLECTION_NAME not in existing_collections:
        vector_size = len(embeddings.embed_query("vector_size_probe"))
        image_vector_size = len(clip_embedder.embed_image(Image.new("RGB",(224,224))))
        image_text_vector_size = len(clip_embedder.embed_text("lala"))
        _client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=vector_size,distance=Distance.COSINE),
                "image_dense":VectorParams(size=image_vector_size,distance=Distance.COSINE),
                "image_text_dense":VectorParams(size=image_text_vector_size,distance=Distance.COSINE)
                },
            sparse_vectors_config= {
                "sparse": SparseVectorParams()
            }
        )

    _vectorstore = QdrantVectorStore(
        client=_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        vector_name="dense",
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID
    )

    return _vectorstore