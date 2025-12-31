from src.pipelines.ingest import ingest
from src.retriever.query_engine import get_retriever
from pprint import pprint

print("Running Ingestion Pipeline from MAIN")
ingest()
retriever = get_retriever()

pprint(retriever.invoke("company"),indent=2)

if '__name__' == '__main__':
    print("Hello world")