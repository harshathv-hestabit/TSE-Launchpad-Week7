# RAG Architecture

Local RAG pipeline for document ingestion, embedding generation, and retrieval using Qdrant vector store.

## Architecture Flow

```
Documents → Load & Clean Documents → Create Chunks (500-800 tokens) → Embed Chunks → Use Qdrant to store vectors → Retrieve documents when query is invoked
```

## Components

### 1. Ingestion Pipeline (`pipelines/ingest.py`)
- Loads PDFs via DirectoryLoader + PyMuPDFLoader
- Cleans text (unicode normalization, whitespace removal)
- Chunks using TokenTextSplitter (690 tokens, no overlap)
- persist intermediate results (pages.jsonl, chunks.jsonl)
- Stores vectors in Qdrant

### 2. Embeddings (`embeddings/embedder.py`)
- HuggingFace embeddings (model via `EMBEDDING_MODEL` env var)
- CPU-based with normalized outputs for cosine similarity

### 3. Vector Store (`vectorstore/vectorstore.py`)
- Qdrant local file storage
- Collection: `embedded_data`
- Distance: COSINE
- Auto-creates collection with dynamic vector dimensions

### 4. Retriever (`retriever/query_engine.py`)
- Similarity search returning top-k chunks (default k=5)
- Returns Document objects with content + metadata

### 5. LLM Client (`generator/llm_client.py`)
- Manages local/API model access
- Provides tokenizer for chunking

## File Structure

```
src/
├── config/model.yaml          # Model configuration
├── data/                      # Contains raw data(pdfs), cleaned data, chunks
├── embeddings/embedder.py     # Embedding wrapper funtion
├── generator/llm_client.py    # LLM interface
├── pipelines/ingest.py        # Ingest pipeline
├── retriever/query_engine.py  # Query interface
└── vectorstore/
    ├── vectorstore.py         # Qdrant integration
    └── qdrant/                # Vector DB storage
```

## Key Design Decisions

- **Token-based chunking**: Ensures consistent chunk sizes for LLM context
- **Two-stage caching**: Separates PDF parsing from chunking for faster iteration
- **Zero overlap**: Prevents duplication, reduces storage
- **Minimal metadata**: Only source and page number preserved
- **Singleton pattern**: Reuses model/client instances

## Environment Variables

```env
EMBEDDING_MODEL=<huggingface-model-id>
DATA_PATH=<raw-documents-path>
CLEAN_PATH=<cleaned-output-path>
CHUNKS_PATH=<chunks-output-path>
```

## Usage Example

```python
from src.pipelines.ingest import ingest
from src.retriever.query_engine import get_retriever

ingest()

retriever = get_retriever(k=5)
results = retriever.invoke("your query")
```