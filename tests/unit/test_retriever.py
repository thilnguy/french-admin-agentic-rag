import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_retrieve_general_returns_results():
    """General domain search should query both collections."""
    with patch(
        "skills.legal_retriever.main._get_qdrant_client"
    ) as mock_client_fn, patch(
        "skills.legal_retriever.main._get_embeddings"
    ) as mock_embed_fn, patch(
        "skills.legal_retriever.main.QdrantVectorStore"
    ) as mock_store_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_fn.return_value = mock_client

        mock_embed_fn.return_value = MagicMock()

        # Mock the vector store search
        mock_doc = MagicMock()
        mock_doc.page_content = "Le passeport coûte 86€"
        mock_doc.metadata = {"title": "Passeport"}

        mock_store = MagicMock()
        mock_store.asimilarity_search = AsyncMock(return_value=[mock_doc])
        mock_store_cls.return_value = mock_store

        from skills.legal_retriever.main import retrieve_legal_info

        results = await retrieve_legal_info("passeport", domain="general")

        assert len(results) > 0
        assert results[0]["source"] in ["service-public", "legi"]
        assert "86€" in results[0]["content"]


@pytest.mark.asyncio
async def test_retrieve_missing_collection_returns_empty():
    """Missing collection should return empty list without error."""
    with patch(
        "skills.legal_retriever.main._get_qdrant_client"
    ) as mock_client_fn, patch(
        "skills.legal_retriever.main._get_embeddings"
    ) as mock_embed_fn:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = False
        mock_client_fn.return_value = mock_client
        mock_embed_fn.return_value = MagicMock()

        from skills.legal_retriever.main import retrieve_legal_info

        results = await retrieve_legal_info("passeport", domain="general")

        assert results == []


@pytest.mark.asyncio
async def test_retrieve_procedure_domain_only():
    """Procedure domain should only query service_public_procedures."""
    with patch(
        "skills.legal_retriever.main._get_qdrant_client"
    ) as mock_client_fn, patch(
        "skills.legal_retriever.main._get_embeddings"
    ) as mock_embed_fn, patch(
        "skills.legal_retriever.main.QdrantVectorStore"
    ) as mock_store_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_fn.return_value = mock_client
        mock_embed_fn.return_value = MagicMock()

        mock_doc = MagicMock()
        mock_doc.page_content = "Procedure info"
        mock_doc.metadata = {"title": "Procedure"}

        mock_store = MagicMock()
        mock_store.asimilarity_search = AsyncMock(return_value=[mock_doc])
        mock_store_cls.return_value = mock_store

        from skills.legal_retriever.main import retrieve_legal_info

        results = await retrieve_legal_info("titre de séjour", domain="procedure")

        # Should only have service-public results
        for r in results:
            assert r["source"] == "service-public"


@pytest.mark.asyncio
async def test_retrieve_legislation_domain_only():
    """Legislation domain should only query legi_legislation."""
    with patch(
        "skills.legal_retriever.main._get_qdrant_client"
    ) as mock_client_fn, patch(
        "skills.legal_retriever.main._get_embeddings"
    ) as mock_embed_fn, patch(
        "skills.legal_retriever.main.QdrantVectorStore"
    ) as mock_store_cls:
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True
        mock_client_fn.return_value = mock_client
        mock_embed_fn.return_value = MagicMock()

        mock_doc = MagicMock()
        mock_doc.page_content = "Article L.123"
        mock_doc.metadata = {"title": "Loi"}

        mock_store = MagicMock()
        mock_store.asimilarity_search = AsyncMock(return_value=[mock_doc])
        mock_store_cls.return_value = mock_store

        from skills.legal_retriever.main import retrieve_legal_info

        results = await retrieve_legal_info("code civil", domain="legislation")

        for r in results:
            assert r["source"] == "legi"
