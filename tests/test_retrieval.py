"""
Tests for hybrid retrieval implementation.
"""

import pytest
from langchain.schema import Document
from src.retrieval import HybridRetriever, create_sample_documents


class TestHybridRetriever:
    """Test cases for HybridRetriever class."""

    @pytest.fixture
    def sample_docs(self):
        """Fixture providing sample documents."""
        return create_sample_documents()

    @pytest.fixture
    def retriever(self, sample_docs):
        """Fixture providing initialized HybridRetriever."""
        return HybridRetriever(
            documents=sample_docs,
            bm25_weight=0.5,
            dense_weight=0.5,
        )

    def test_initialization(self, sample_docs):
        """Test retriever initializes correctly."""
        retriever = HybridRetriever(documents=sample_docs)
        assert retriever is not None
        assert retriever.bm25_weight == 0.5
        assert retriever.dense_weight == 0.5

    def test_exact_keyword_match(self, retriever):
        """Test BM25 catches exact keyword matches."""
        query = "503-AUTH-TIMEOUT"
        results = retriever.get_relevant_documents(query, top_k=3)

        assert len(results) > 0
        # First result should contain the error code
        assert "503-AUTH-TIMEOUT" in results[0].page_content

    def test_semantic_search(self, retriever):
        """Test dense retrieval handles semantic queries."""
        query = "How do I fix authentication timeouts?"
        results = retriever.get_relevant_documents(query, top_k=3)

        assert len(results) > 0
        # Should retrieve docs about authentication or timeouts
        relevant_terms = ["authentication", "timeout", "auth"]
        assert any(
            term.lower() in results[0].page_content.lower()
            for term in relevant_terms
        )

    def test_top_k_limiting(self, retriever):
        """Test top_k parameter limits results."""
        query = "authentication"
        results = retriever.get_relevant_documents(query, top_k=2)

        assert len(results) <= 2

    def test_weight_update(self, retriever):
        """Test dynamic weight updates."""
        retriever.update_weights(bm25_weight=0.7, dense_weight=0.3)

        assert retriever.bm25_weight == 0.7
        assert retriever.dense_weight == 0.3


class TestSampleDocuments:
    """Test sample document creation."""

    def test_create_sample_documents(self):
        """Test sample documents are created correctly."""
        docs = create_sample_documents()

        assert len(docs) > 0
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(doc.page_content for doc in docs)
        assert all("doc_id" in doc.metadata for doc in docs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
