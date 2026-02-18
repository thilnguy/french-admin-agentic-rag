import asyncio
from functools import lru_cache
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from src.config import settings
from src.utils.logger import logger
from src.utils import metrics
from src.shared.reranker import get_reranker
import time


# Singleton clients — avoid re-creating expensive connections per request
@lru_cache(maxsize=1)
def _get_qdrant_client():
    logger.info("Initializing Qdrant client...")
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


@lru_cache(maxsize=1)
def _get_embeddings():
    logger.info("Loading HuggingFace embeddings model (BAAI/bge-m3)...")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-m3")


async def retrieve_legal_info(query: str, domain: str = "general", user_profile=None):
    """
    Retrieves information about French administrative procedures or legislation.
    domain: 'procedure' (service-public) or 'legislation' (legi) or 'general' (both)
    """
    client = _get_qdrant_client()
    embeddings = _get_embeddings()

    async def search_collection(collection_name, label, k):
        if not client.collection_exists(collection_name):
            return []
        store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
            content_payload_key="text",
        )
        docs = await store.asimilarity_search(query, k=k)
        return [
            {"source": label, "content": d.page_content, "metadata": d.metadata}
            for d in docs
        ]

    search_tasks = []
    if domain in ["procedure", "general"]:
        search_tasks.append(
            search_collection("service_public_procedures", "service-public", 6)
        )
    if domain in ["legislation", "general"]:
        search_tasks.append(search_collection("legi_legislation", "legi", 4))

    if not search_tasks:
        return []

    start_time = time.time()
    batch_results = await asyncio.gather(*search_tasks)
    duration = time.time() - start_time
    metrics.RAG_RETRIEVAL_LATENCY.labels(domain=domain).observe(duration)

    results = []
    for batch in batch_results:
        results.extend(batch)

    logger.debug(f"Retriever found {len(results)} results for query: '{query}'")
    for r in results:
        logger.debug(
            f" - Found: {r['source']} | Title: {r['metadata'].get('title', 'N/A')}"
        )

    # Context-Aware Reranking (Layer 3)
    reranker = get_reranker()
    reranked_results = reranker.rerank(query, results, user_profile=user_profile)
    
    logger.debug(f"Reranked top {len(reranked_results)} results.")

    return reranked_results


def warmup():
    """Pre-load singleton clients on startup."""
    _get_qdrant_client()
    _get_embeddings()


if __name__ == "__main__":
    # Example usage / test
    # asyncio.run(retrieve_legal_info("Comment faire un passeport ?"))
    pass
