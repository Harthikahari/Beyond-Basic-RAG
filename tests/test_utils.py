"""
Tests for utility functions.
"""

import pytest
from langchain.schema import Document
from src.utils import (
    reorder_documents_lost_in_middle,
    chunk_documents,
    deduplicate_documents,
    format_documents_for_prompt,
    calculate_context_stats,
)


class TestDocumentReordering:
    """Test document reordering functions."""

    def test_reorder_five_documents(self):
        """Test reordering with 5 documents."""
        docs = [
            Document(page_content=f"Doc {i}", metadata={"id": i})
            for i in range(1, 6)
        ]

        reordered = reorder_documents_lost_in_middle(docs)

        # Expected order: [1, 3, 5, 4, 2]
        expected_ids = [1, 3, 5, 4, 2]
        actual_ids = [doc.metadata["id"] for doc in reordered]

        assert actual_ids == expected_ids

    def test_reorder_two_documents(self):
        """Test reordering with 2 documents (no change expected)."""
        docs = [
            Document(page_content="Doc 1", metadata={"id": 1}),
            Document(page_content="Doc 2", metadata={"id": 2}),
        ]

        reordered = reorder_documents_lost_in_middle(docs)

        assert len(reordered) == 2
        assert reordered[0].metadata["id"] == 1
        assert reordered[1].metadata["id"] == 2

    def test_reorder_empty_list(self):
        """Test reordering with empty list."""
        docs = []
        reordered = reorder_documents_lost_in_middle(docs)
        assert reordered == []


class TestChunkDocuments:
    """Test document chunking."""

    def test_chunk_long_document(self):
        """Test chunking a long document."""
        long_text = "This is a sentence. " * 100
        docs = [Document(page_content=long_text, metadata={"doc_id": "test"})]

        chunks = chunk_documents(docs, chunk_size=200, chunk_overlap=20)

        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 250 for chunk in chunks)
        assert all("chunk_index" in chunk.metadata for chunk in chunks)

    def test_chunk_preserves_metadata(self):
        """Test that chunking preserves original metadata."""
        doc = Document(
            page_content="Long text " * 100,
            metadata={"doc_id": "test", "category": "example"},
        )

        chunks = chunk_documents([doc], chunk_size=100)

        assert all(chunk.metadata["doc_id"] == "test" for chunk in chunks)
        assert all(chunk.metadata["category"] == "example" for chunk in chunks)


class TestDeduplication:
    """Test document deduplication."""

    def test_deduplicate_identical(self):
        """Test deduplication of identical documents."""
        docs = [
            Document(page_content="Same content", metadata={"id": 1}),
            Document(page_content="Same content", metadata={"id": 2}),
            Document(page_content="Different content", metadata={"id": 3}),
        ]

        unique = deduplicate_documents(docs, similarity_threshold=0.95)

        assert len(unique) == 2

    def test_deduplicate_all_unique(self):
        """Test deduplication with all unique documents."""
        docs = [
            Document(page_content=f"Content {i}", metadata={"id": i})
            for i in range(5)
        ]

        unique = deduplicate_documents(docs)

        assert len(unique) == 5


class TestFormatDocuments:
    """Test document formatting."""

    def test_format_basic(self):
        """Test basic document formatting."""
        docs = [
            Document(page_content="Content 1", metadata={"id": 1}),
            Document(page_content="Content 2", metadata={"id": 2}),
        ]

        formatted = format_documents_for_prompt(docs)

        assert "Document 1:" in formatted
        assert "Document 2:" in formatted
        assert "Content 1" in formatted
        assert "Content 2" in formatted

    def test_format_with_scores(self):
        """Test formatting with rerank scores."""
        docs = [
            Document(
                page_content="Content 1",
                metadata={"id": 1, "rerank_score": 0.95},
            ),
        ]

        formatted = format_documents_for_prompt(
            docs, include_scores=True
        )

        assert "0.95" in formatted or "Relevance" in formatted


class TestContextStats:
    """Test context statistics calculation."""

    def test_calculate_stats(self):
        """Test basic statistics calculation."""
        docs = [
            Document(page_content="Short", metadata={}),
            Document(page_content="A bit longer text", metadata={}),
            Document(page_content="Even longer text here", metadata={}),
        ]

        stats = calculate_context_stats(docs)

        assert stats["num_documents"] == 3
        assert stats["total_chars"] > 0
        assert stats["total_tokens"] > 0
        assert stats["avg_doc_length"] > 0

    def test_calculate_stats_with_scores(self):
        """Test statistics with rerank scores."""
        docs = [
            Document(
                page_content="Text",
                metadata={"rerank_score": 0.9},
            ),
            Document(
                page_content="Text",
                metadata={"rerank_score": 0.7},
            ),
        ]

        stats = calculate_context_stats(docs)

        assert stats["has_scores"] is True
        assert stats["avg_score"] == 0.8
        assert stats["min_score"] == 0.7
        assert stats["max_score"] == 0.9

    def test_calculate_stats_empty(self):
        """Test statistics with empty list."""
        stats = calculate_context_stats([])

        assert stats["num_documents"] == 0
        assert stats["total_chars"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
