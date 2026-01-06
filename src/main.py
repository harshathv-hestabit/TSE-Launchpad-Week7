from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import yaml
import os

from huggingface_hub import InferenceClient
from src.pipelines.ingest import ingest
from src.retriever.query_engine import get_retriever
from src.retriever.hybrid_retriever import hybrid_retrieve
from src.pipelines.context_builder import ContextBuilder
from generator.llm_client import LLM

# print("Running Ingestion Pipeline from MAIN")
# ingest()
# retriever = get_retriever()

# pprint(retriever.invoke("company"),indent=2)

# results = hybrid_retrieve(query="What was Downer's total revenue for 2022?")

# print("\n Context Builder results below")
# contextBuilder = ContextBuilder()
# context = contextBuilder.build(documents=results)
# print(context)


'''
Generator test
'''

# with open(Path("src/config/model.yaml"), "r") as f:
#     config = yaml.safe_load(f)

# llm = LLM(config)
# model = llm.get_model()

# prompt = "Explain RAG in one paragraph."

# if isinstance(model, InferenceClient):
#     messages = [
#         {
#             "role": "system",
#             "content": "You are a helpful assistant."
#         },
#         {
#             "role": "user",
#             "content": prompt
#         }
#     ]

#     response = model.chat_completion(
#         messages=messages,
#         max_tokens=200,
#         temperature=0.2,
#     )

#     # output = response["choices"][0]["message"]["content"]
#     output = response
# print(output)
from pathlib import Path
from PIL import Image
from retriever.image_search import ImageSearcher

COLLECTION = "embedded_data"

searcher = ImageSearcher()

print("\nTEXT → IMAGE")
results = searcher.search_by_text(
    query="organizational hierarchy chart",
    limit=3,
)

if results:
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score: {r['score']}")
        if r['page_content']:
            print(f"   Content: {r['page_content'][:150]}...")
else:
    print("No results found for text query")

print("\n\nIMAGE → IMAGE")
test_image = Path("src/data/cleaned/images/figure-1-1.jpg")

if test_image.exists():
    results = searcher.search_by_image(test_image, limit=3)
    
    if results:
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Score: {r['score']}")
    else:
        print("No results found for image query")
else:
    print(f"Test image not found: {test_image}")

print("\n\nIMAGE → TEXT")
if test_image.exists():
    results = searcher.search_image_to_text(test_image, limit=3)
    
    if results:
        for i, r in enumerate(results, 1):
            print(f"\n{i}. Score: {r['score']}")
            if r['page_content']:
                print(f"   Content: {r['page_content'][:150]}...")
    else:
        print("No results found for image-to-text query")
else:
    print(f"Test image not found: {test_image}")