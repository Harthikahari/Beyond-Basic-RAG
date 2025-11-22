"""
Tests for cross-encoder reranking implementation.
"""

import pytest
from langchain.schema import Document
from src.reranker import CrossEncoderReranker, RerankedRetriever
from src.retrieval import HybridRetriever, create_sample_documents


class TestCrossEncoderReranker:
    """Test cases for CrossEncoderReranker class."""

    @pytest.fixture
    def reranker(self):
        """Fixture providing initialized reranker."""
        return CrossEncoderReranker(model_name="fast")

    @pytest.fixture
    def sample_docs(self):
        """Fixture providing sample documents."""
        return create_sample_documents()

    def test_initialization(self):
        """Test reranker initializes correctly."""
        reranker = CrossEncoderReranker("fast")
        assert reranker is not None
        assert "ms-marco-MiniLM" in reranker.model_name

    def test_model_shortcuts(self):
        """Test model name shortcuts work."""
        shortcuts = ["fast", "accurate", "qa", "multilingual"]
        for shortcut in shortcuts:
            reranker = CrossEncoderReranker(shortcut)
            assert reranker is not None

    def test_rerank_basic(self, reranker, sample_docs):
        """Test basic reranking functionality."""
        query = "How do I fix authentication timeouts?"
        reranked = reranker.rerank(query, sample_docs, top_k=3)

        assert len(reranked) == 3
        assert all(isinstance(doc, Document) for doc in reranked)

    def test_rerank_scores_added(self, reranker, sample_docs):
        """Test that rerank scores are added to metadata."""
        query = "authentication timeout"
        reranked = reranker.rerank(
            query, sample_docs, top_k=3, return_scores=True
        )

        assert all("rerank_score" in doc.metadata for doc in reranked)
        assert all(isinstance(doc.metadata["rerank_score"], float) for doc in reranked)

    def test_rerank_ordering(self, reranker, sample_docs):
        """Test that results are ordered by score."""
        query = "authentication timeout"
        reranked = reranker.rerank(query, sample_docs, top_k=5)

        scores = [doc.metadata["rerank_score"] for doc in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_with_threshold(self, reranker, sample_docs):
        """Test threshold-based filtering."""
        query = "authentication timeout"
        filtered = reranker.rerank_with_threshold(
            query, sample_docs, score_threshold=0.0, max_results=3
        )

        assert len(filtered) <= 3
        assert all(doc.metadata["rerank_score"] >= 0.0 for doc in filtered)

    def test_score_pairs(self, reranker, sample_docs):
        """Test scoring without reordering."""
        query = "error 503"
        scored = reranker.score_pairs(query, sample_docs)

        assert len(scored) == len(sample_docs)
        assert all(isinstance(pair, tuple) for pair in scored)
        assert all(len(pair) == 2 for pair in scored)

    def test_score_statistics(self, reranker, sample_docs):
        """Test score statistics calculation."""
        query = "authentication"
        stats = reranker.get_score_statistics(query, sample_docs)

        assert "mean" in stats
        assert "median" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert stats["count"] == len(sample_docs)


class TestRerankedRetriever:
    """Test cases for RerankedRetriever wrapper."""

    @pytest.fixture
    def retriever(self):
        """Fixture providing complete reranked retriever."""
        docs = create_sample_documents()
        hybrid = HybridRetriever(docs)
        reranker = CrossEncoderReranker("fast")

        return RerankedRetriever(
            base_retriever=hybrid,
            reranker=reranker,
            top_k=3,
            apply_context_reordering=True,
        )

    def test_end_to_end_retrieval(self, retriever):
        """Test complete retrieval pipeline."""
        query = "How do I fix authentication timeouts?"
        results = retriever.get_relevant_documents(query)

        assert len(results) <= 3
        assert all(isinstance(doc, Document) for doc in results)
        assert all("rerank_score" in doc.metadata for doc in results)

    def test_context_reordering_applied(self, retriever):
        """Test that context reordering is applied."""
        query = "authentication timeout"
        results = retriever.get_relevant_documents(query)

        # With reordering, second-best doc should be at the end
        assert len(results) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
