from dotenv import load_dotenv
load_dotenv()
from vectorstore.vectorstore import get_vectorstore

def get_retriever(k: int = 5):
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever