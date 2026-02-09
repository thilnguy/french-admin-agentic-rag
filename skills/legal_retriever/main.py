import os
import antigravity as ag
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

# Load environment variables in main.py usually, but skills might need their own setup if tested in isolation
# from dotenv import load_dotenv
# load_dotenv()

@ag.skill(name="legal_retriever")
def retrieve_legal_info(query: str, domain: str = "general"):
    """
    Retrieves information about French administrative procedures or legislation.
    domain: 'procedure' (service-public) or 'legislation' (legi) or 'general' (both)
    """
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    results = []
    
    # Logic to search in specific collections
    if domain in ["procedure", "general"]:
        store_proc = QdrantVectorStore(
            client=client, 
            collection_name="service_public_procedures", 
            embeddings=embeddings
        )
        proc_docs = store_proc.similarity_search(query, k=3)
        results.extend([{"source": "service-public", "content": doc.page_content, "metadata": doc.metadata} for doc in proc_docs])
        
    if domain in ["legislation", "general"]:
        store_legi = QdrantVectorStore(
            client=client, 
            collection_name="legi_legislation", 
            embeddings=embeddings
        )
        legi_docs = store_legi.similarity_search(query, k=2)
        results.extend([{"source": "legi", "content": doc.page_content, "metadata": doc.metadata} for doc in legi_docs])
        
    return results

if __name__ == "__main__":
    # Example usage / test
    # print(retrieve_legal_info("Comment faire un passeport ?"))
    pass
