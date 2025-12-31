from dotenv import load_dotenv
load_dotenv()

import os
import json
import unicodedata
import re

from pathlib import Path

# Document Loaders (dirctory of .pdf files)
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyMuPDFLoader

# Text splitters based on token count
from langchain_text_splitters import TokenTextSplitter

# Using Document Wrapper for JSONL read and write operations
from langchain_core.documents import Document

# loading config variables
import yaml
from generator.llm_client import LLM
from vectorstore.vectorstore import get_vectorstore

# Getting Config values
with open(Path("src/config/model.yaml"), "r") as f:
    config = yaml.safe_load(f)

# Defining Paths
data_path = Path(os.getenv('DATA_PATH'))
cleaned_path = Path(os.getenv('CLEAN_PATH'))
chunks_path = Path(os.getenv('CHUNKS_PATH'))

CLEANED_FILE = cleaned_path / "pages.jsonl"
CHUNKS_FILE = chunks_path / "chunks.jsonl"

pdf_pattern = "**/*.pdf"

def ingest():
    
    '''
    Check for cleaned file in cleaned directory. Cleaned file refer to documents loaded in page mode.
    The cleaned file is in JSONL format.
    If no cleaned file exists then use Directory Loader with `PyMuPDFLoader` to process documents in page mode, matching only pdfs
    '''
    print("Cleaning data")
    
    if CLEANED_FILE.exists():
        print("Loading cleaned documents")
        documents = []
        with open(CLEANED_FILE, "r",encoding='UTF-8') as f:
            for line in f:
                obj = json.loads(line)
                documents.append(
                    Document(
                        page_content= obj["content"],
                        metadata = obj["metadata"]
                    )
                )
    else:
        print("Using Directory Loader to load documents")
        
        loader = DirectoryLoader(
            path=data_path,
            glob=pdf_pattern,
            recursive= True,
            loader_cls= PyMuPDFLoader,
            loader_kwargs={
                "mode":"page"
                },
            show_progress= True)

        documents = loader.load()

        for document in documents:
            text = document.page_content
            text = unicodedata.normalize("NFKC", text)
            text = re.sub(r"\n{2,}", "\n", text)
            text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]", "", text)
            document.page_content = text.strip()
        
        with open(CLEANED_FILE, "w", encoding="UTF-8") as f:
            for document in documents:
                f.write(
                    json.dumps(
                        {
                            "content":document.page_content,
                            "metadata":document.metadata
                        },
                        ensure_ascii=False
                    )
                    + '\n'
                )

    print(f"Total number of pages loaded from documents after cleaning: {len(documents)}")

    '''
    Now that the documents have been loaded, we will use TokenTextSplitter to split each page into chunks 
    having a custom token size ranging from 500 to 800 tokens per chunk
    '''
    
    print("Creating Chunks from cleaned data")
    if CHUNKS_FILE.exists():
        print("Loading processed chunks")
        chunks = []
        with open(CHUNKS_FILE, "r", encoding="UTF-8") as f:
            for line in f:
                obj = json.loads(line)
                chunks.append(
                    Document(
                        page_content = obj["content"],
                        metadata = obj["metadata"]
                    )
                )
    else:
        print("Processing documents into chunks")
        llm = LLM(config)
        model_tokenizer = llm.get_tokenizer()
        text_splitter = TokenTextSplitter.from_huggingface_tokenizer(tokenizer=model_tokenizer,chunk_size=690,chunk_overlap=0)

        chunks = text_splitter.split_documents(documents)
        
        with open(CHUNKS_FILE, "w", encoding="UTF-8") as f:
            for chunk in chunks:
                f.write(
                    json.dumps(
                        {
                            "content":chunk.page_content,
                            "metadata":chunk.metadata
                        },
                        ensure_ascii=False
                    )
                    + '\n'
                )

    print(f"Total number of chunks created: {len(chunks)}")
    
    '''
    Modifying the chunk property `metadata` to contain only source and page number properties.
    '''
    
    print("Processing chunks to modify metadata object")
    print(f"Sample metadata content for a chunk: \n{chunks[0].metadata}")
    for chunk in chunks:
        chunk.metadata = {
            "source": chunk.metadata.get('source'),
            "page": chunk.metadata.get('page')
        }
    
    print(f"Modified Metadata Object Value of a sample chunks:\n")
    for i,chunk in enumerate(chunks[:10]):
        print(f"{i}: {chunk.metadata}",end='\n')
        
    '''
    Add processed chunks into the vectorstore
    '''

    print("Adding chunks to vectorstore")

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    print(f"Successfully added {len(chunks)} chunks to vectorstore")

        
if __name__ == '__main__':
    print("Running Ingestion Pipeline")
    ingest()