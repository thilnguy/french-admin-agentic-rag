import os
import asyncio
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.config import settings
from src.utils.logger import logger

async def retrieve_legal_info(query: str, domain: str = "general"):
    """
    Retrieves information about French administrative procedures or legislation.
    domain: 'procedure' (service-public) or 'legislation' (legi) or 'general' (both)
    """
    # Use centralized settings
    client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)
    
    # Matching the embeddings used in AgentPublic datasets (BGE-M3)
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    results = []
    
    # Logic to search in specific collections with survival check
    # QdrantVectorStore.asimilarity_search is available but QdrantClient might be sync.
    # We will use run_in_executor to avoid blocking the event loop if strictly needed,
    # but langchain's asimilarity_search handles some async logic. 
    # To be safe and since we create a new store each time, we'll wrap the sync calls or use asimilarity_search where possible.
    
    tasks = []

    if domain in ["procedure", "general"]:
        if client.collection_exists("service_public_procedures"):
            store_proc = QdrantVectorStore(
                client=client, 
                collection_name="service_public_procedures", 
                embedding=embeddings,
                content_payload_key="text"
            )
            tasks.append(store_proc.asimilarity_search(query, k=3))
            
    if domain in ["legislation", "general"]:
        if client.collection_exists("legi_legislation"):
            store_legi = QdrantVectorStore(
                client=client, 
                collection_name="legi_legislation", 
                embedding=embeddings,
                content_payload_key="text"
            )
            tasks.append(store_legi.asimilarity_search(query, k=2))
            
    if not tasks:
        return []

    # Run searches in parallel
    search_results = await asyncio.gather(*tasks)
    
    # Flatten results
    for i, docs in enumerate(search_results):
        # Determine source provided we appended in order. 
        # This implementation is slightly naive about source attribution if one collection is missing.
        # Improved logic:
        source_label = "unknown"
        # We need to preserve mapping.
        pass 

    # Re-implementation for robust source mapping
    results = []
    
    # Helper for specific search
    async def search_collection(collection_name, label, k):
        if not client.collection_exists(collection_name):
            return []
        store = QdrantVectorStore(
            client=client, 
            collection_name=collection_name, 
            embedding=embeddings,
            content_payload_key="text"
        )
        docs = await store.asimilarity_search(query, k=k)
        return [{"source": label, "content": d.page_content, "metadata": d.metadata} for d in docs]

    search_tasks = []
    if domain in ["procedure", "general"]:
        search_tasks.append(search_collection("service_public_procedures", "service-public", 3))
    if domain in ["legislation", "general"]:
        search_tasks.append(search_collection("legi_legislation", "legi", 2))
        
    batch_results = await asyncio.gather(*search_tasks)
    for batch in batch_results:
        results.extend(batch)
        
    logger.debug(f"Retriever found {len(results)} results for query: '{query}'")
    for r in results:
        content_snippet = r['content'][:200].replace('\n', ' ')
        logger.debug(f" - Found: {r['source']} | Title: {r['metadata'].get('title', 'N/A')}")
        
    return results

if __name__ == "__main__":
    # Example usage / test
    # asyncio.run(retrieve_legal_info("Comment faire un passeport ?"))
    pass
