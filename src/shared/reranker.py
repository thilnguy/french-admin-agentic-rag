from sentence_transformers import CrossEncoder
from typing import List, Dict, Any, Optional
from src.utils.logger import logger
from functools import lru_cache


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Initializing Reranker with model: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        user_profile: Optional[Any] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of documents based on relevance to the query + user context.
        Docs must be a list of dicts with 'content' key.
        """
        if not docs:
            return []

        # Prepare pairs for Cross-Encoder
        # If user_profile is provided, enrich the query with context
        augmented_query = query
        if user_profile:
            context_parts = []
            if getattr(user_profile, "valid", True):  # Duck checking or Pydantic
                if getattr(user_profile, "location", None):
                    context_parts.append(f"User Location: {user_profile.location}")
                if getattr(user_profile, "nationality", None):
                    context_parts.append(
                        f"User Nationality: {user_profile.nationality}"
                    )
                if getattr(user_profile, "residency_status", None):
                    context_parts.append(
                        f"User Status: {user_profile.residency_status}"
                    )

            if context_parts:
                augmented_query = f"{query} [Context: {', '.join(context_parts)}]"

        # (Query, Document Content)
        pairs = [(augmented_query, doc["content"]) for doc in docs]

        # Predict scores
        scores = self.model.predict(pairs)

        # Attach scores to docs
        for i, doc in enumerate(docs):
            doc["score"] = float(scores[i])

        # Sort by score descending
        ranked_docs = sorted(docs, key=lambda x: x["score"], reverse=True)

        return ranked_docs[:top_k]


# Singleton
@lru_cache(maxsize=1)
def get_reranker():
    return Reranker()
