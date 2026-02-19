"""
HybridRetriever — combines Qdrant semantic search with BM25 lexical search
using Reciprocal Rank Fusion (RRF).

DESIGN:
  The BM25 index is built on-the-fly from the top-K Qdrant candidates.
  This avoids maintaining a persistent corpus while still improving recall
  for keyword-heavy queries that the dense embedder may miss.

  Reciprocal Rank Fusion formula (Cormack et al., 2009):
      RRF_score(doc) = Σ  1 / (k + rank_i)
  where k=60 is the standard constant that dampens the influence of rank.

  Final results are sorted by descending RRF score and truncated to top_n.
"""

from __future__ import annotations

import re
from typing import Any

_BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover
    _BM25_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_STOP_WORDS_FR = frozenset(
    [
        "le",
        "la",
        "les",
        "de",
        "du",
        "des",
        "un",
        "une",
        "et",
        "en",
        "à",
        "au",
        "aux",
        "est",
        "pour",
        "par",
        "sur",
        "ce",
        "je",
        "il",
        "elle",
        "on",
        "nous",
        "vous",
        "ils",
        "elles",
        "que",
        "qui",
        "ou",
        "si",
        "ne",
        "pas",
        "plus",
        "avec",
        "dans",
    ]
)


def _tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumeric, remove short tokens + stop words."""
    tokens = re.findall(r"[a-zA-Z0-9\u00C0-\u024F]+", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOP_WORDS_FR]


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

_RRF_K = 60  # Standard constant from Cormack et al. 2009


def _rrf_merge(
    ranked_lists: list[list[int]],
    n_docs: int,
    top_n: int,
) -> list[int]:
    """
    Merge multiple ranked lists of doc indices using RRF.

    Args:
        ranked_lists: Each list is an ordered sequence of document indices
                      (best to worst).
        n_docs:       Total number of documents.
        top_n:        Number of results to return.

    Returns:
        Ordered list of document indices (best first), length <= top_n.
    """
    scores: dict[int, float] = {i: 0.0 for i in range(n_docs)}
    for ranked_list in ranked_lists:
        for rank, doc_idx in enumerate(ranked_list):
            scores[doc_idx] += 1.0 / (_RRF_K + rank + 1)

    # Sort by descending score
    ordered = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)
    return ordered[:top_n]


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------


class HybridRetriever:
    """
    Wraps a list of candidate documents and fuses Qdrant semantic ordering
    with a BM25 lexical ranking via Reciprocal Rank Fusion.

    Usage:
        retriever = HybridRetriever(qdrant_results)
        fused_results = retriever.rerank(query, top_n=10)
    """

    def __init__(self, docs: list[dict[str, Any]]):
        """
        Args:
            docs: Documents in Qdrant semantic order (best first).
                  Each doc must have a 'content' key.
        """
        self._docs = docs
        self._bm25: BM25Okapi | None = None

        if not docs or not _BM25_AVAILABLE:
            return  # Graceful degradation — BM25 unavailable

        corpus = [_tokenize(d.get("content", "")) for d in docs]
        if any(corpus):
            self._bm25 = BM25Okapi(corpus)

    def rerank(self, query: str, top_n: int = 10) -> list[dict[str, Any]]:
        """
        Fuse Qdrant semantic rank and BM25 rank using RRF.

        Args:
            query:  The user query (used for BM25 scoring).
            top_n:  Maximum number of results to return.

        Returns:
            Fused and re-ranked documents (best first).
        """
        docs = self._docs
        if not docs:
            return []

        n = len(docs)

        # Qdrant order (already ranked 0..n-1 best to worst)
        qdrant_rank = list(range(n))

        if self._bm25 is None:
            # BM25 unavailable — return semantic ordering unchanged
            return docs[:top_n]

        # BM25 ordering
        query_tokens = _tokenize(query)
        if not query_tokens:
            # Empty query tokens → semantic ordering wins
            return docs[:top_n]

        bm25_scores = self._bm25.get_scores(query_tokens)
        bm25_rank = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)

        # RRF fusion
        fused_indices = _rrf_merge([qdrant_rank, bm25_rank], n_docs=n, top_n=top_n)
        return [docs[i] for i in fused_indices]


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def hybrid_rerank(
    documents: list[dict[str, Any]],
    query: str,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """
    Convenience function: build a HybridRetriever and rerank in one call.

    If rank_bm25 is unavailable, silently falls back to the input ordering.
    """
    return HybridRetriever(documents).rerank(query, top_n=top_n)
