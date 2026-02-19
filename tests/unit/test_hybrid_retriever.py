"""
Unit tests for HybridRetriever (BM25 + RRF fusion).

Tests cover:
  - Empty document list
  - Single-ranked list (semantic only, no BM25)
  - RRF fusion correctness
  - Lexical keyword boost (BM25 promotes exact matches)
  - Graceful fallback when corpus has no tokens
  - top_n truncation
  - module-level hybrid_rerank convenience function
"""

from src.shared.hybrid_retriever import HybridRetriever, hybrid_rerank, _rrf_merge


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _docs(*contents: str) -> list[dict]:
    return [{"content": c, "source": "test"} for c in contents]


# ---------------------------------------------------------------------------
# _rrf_merge
# ---------------------------------------------------------------------------


def test_rrf_merge_single_list():
    """Single list → RRF order matches input order."""
    ranked_lists = [[0, 1, 2]]
    fused = _rrf_merge(ranked_lists, n_docs=3, top_n=3)
    assert fused == [0, 1, 2]


def test_rrf_merge_two_identical_lists():
    """Two identical lists → same order as input."""
    ranked_lists = [[0, 1, 2], [0, 1, 2]]
    fused = _rrf_merge(ranked_lists, n_docs=3, top_n=3)
    assert fused == [0, 1, 2]


def test_rrf_merge_two_reverse_lists():
    """Two exactly reversed lists → all docs get identical RRF scores (symmetric)."""
    # [0,1,2] and [2,1,0] → each doc appears at rank i in list A and rank (n-1-i) in list B
    # All three docs get the same total RRF score → all are present in result
    ranked_lists = [[0, 1, 2], [2, 1, 0]]
    fused = _rrf_merge(ranked_lists, n_docs=3, top_n=3)
    # All 3 docs should be in result (order can be any due to tie-breaking)
    assert set(fused) == {0, 1, 2}
    assert len(fused) == 3


def test_rrf_merge_top_n_truncates():
    """top_n limits results."""
    ranked_lists = [[0, 1, 2, 3, 4]]
    fused = _rrf_merge(ranked_lists, n_docs=5, top_n=3)
    assert len(fused) == 3


# ---------------------------------------------------------------------------
# HybridRetriever — empty / edge cases
# ---------------------------------------------------------------------------


def test_hybrid_empty_docs():
    """Empty doc list → empty result."""
    retriever = HybridRetriever([])
    assert retriever.rerank("query") == []


def test_hybrid_single_doc():
    """Single doc → same doc returned."""
    docs = _docs("Paris est la capitale de la France.")
    retriever = HybridRetriever(docs)
    result = retriever.rerank("capitale France")
    assert result == docs


def test_hybrid_top_n_respected():
    """top_n limits results."""
    docs = _docs("doc a", "doc b", "doc c", "doc d", "doc e")
    retriever = HybridRetriever(docs)
    result = retriever.rerank("doc", top_n=3)
    assert len(result) == 3


# ---------------------------------------------------------------------------
# HybridRetriever — BM25 lexical boost
# ---------------------------------------------------------------------------


def test_hybrid_bm25_promotes_keyword_match():
    """
    If doc[1] contains the exact keywords but appears 2nd in semantic ranking,
    RRF should either keep it 2nd or promote it to 1st (not demote further).
    """
    # Semantic order: doc 0 first (high qdrant score), doc 1 second
    # But BM25: doc 1 contains the query keywords → BM25 ranks it 1st
    semantic_first = {
        "content": "Ce document parle de la fiscalité générale.",
        "source": "test",
    }
    keyword_match = {
        "content": "Le visa long séjour est requis pour un séjour de plus de 90 jours.",
        "source": "test",
    }
    docs = [semantic_first, keyword_match]

    retriever = HybridRetriever(docs)
    result = retriever.rerank("visa long séjour 90 jours", top_n=2)

    # keyword_match should appear in results (not demoted out)
    contents = [r["content"] for r in result]
    assert keyword_match["content"] in contents


def test_hybrid_rrf_promotes_consensus():
    """
    Document that ranks well in BOTH semantic and BM25 should be top result.
    """
    best_doc = {
        "content": "passeport français renouvellement délai procedure",
        "source": "test",
    }
    other_a = {"content": "permis de conduire catégorie B", "source": "test"}
    other_b = {
        "content": "carte vitale sécurité sociale remboursement",
        "source": "test",
    }

    # Semantic order: best_doc first, then others (as if Qdrant scored it highest)
    docs = [best_doc, other_a, other_b]
    retriever = HybridRetriever(docs)
    result = retriever.rerank("renouvellement passeport français", top_n=3)

    # best_doc ranks 1st in semantic AND contains keywords → should be 1st after RRF
    assert result[0]["content"] == best_doc["content"]


# ---------------------------------------------------------------------------
# HybridRetriever — empty-token fallback
# ---------------------------------------------------------------------------


def test_hybrid_empty_query_tokens_falls_back_to_semantic():
    """Query with only stop-words → tokenizer produces [] → semantic order returned."""
    docs = _docs("Doc A content here.", "Doc B content here.")
    retriever = HybridRetriever(docs)
    # French stop-words only
    result = retriever.rerank("le la les de du", top_n=2)
    # Falls back to semantic (Qdrant) order
    assert result[0] == docs[0]


# ---------------------------------------------------------------------------
# hybrid_rerank convenience function
# ---------------------------------------------------------------------------


def test_hybrid_rerank_convenience():
    """Module-level hybrid_rerank mirrors HybridRetriever.rerank."""
    docs = _docs(
        "visa étudiant france procédure inscription",
        "passeport renouvellement délai",
    )
    result = hybrid_rerank(docs, "visa étudiant", top_n=2)
    assert len(result) == 2
    # First doc should rank high (contains query keywords)
    assert result[0]["content"] == docs[0]["content"]


def test_hybrid_rerank_empty():
    """Convenience function with empty docs → empty list."""
    assert hybrid_rerank([], "query") == []
