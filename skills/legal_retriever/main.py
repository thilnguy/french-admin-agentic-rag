import os
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

def retrieve_legal_info(query: str, domain: str = "general"):
    """
    Retrieves information about French administrative procedures or legislation.
    domain: 'procedure' (service-public) or 'legislation' (legi) or 'general' (both)
    """
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", 6333)))
    
    # Matching the embeddings used in AgentPublic datasets (BGE-M3)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    results = []
    
    # Logic to search in specific collections with survival check
    if domain in ["procedure", "general"]:
        if client.collection_exists("service_public_procedures"):
            store_proc = QdrantVectorStore(
                client=client, 
                collection_name="service_public_procedures", 
                embedding=embeddings,
                content_payload_key="text"
            )
            proc_docs = store_proc.similarity_search(query, k=3)
            results.extend([{"source": "service-public", "content": doc.page_content, "metadata": doc.metadata} for doc in proc_docs])
        
    if domain in ["legislation", "general"]:
        if client.collection_exists("legi_legislation"):
            store_legi = QdrantVectorStore(
                client=client, 
                collection_name="legi_legislation", 
                embedding=embeddings,
                content_payload_key="text"
            )
            legi_docs = store_legi.similarity_search(query, k=2)
            results.extend([{"source": "legi", "content": doc.page_content, "metadata": doc.metadata} for doc in legi_docs])
        
    print(f"DEBUG: Retriever found {len(results)} results for query: '{query}'")
    for r in results:
        content_snippet = r['content'][:200].replace('\n', ' ')
        print(f" - Found: {r['source']} | Title: {r['metadata'].get('title', 'N/A')}")
        print(f"   Snippet: {content_snippet}...")
        
    return results

if __name__ == "__main__":
    # Example usage / test
    # print(retrieve_legal_info("Comment faire un passeport ?"))
    pass
