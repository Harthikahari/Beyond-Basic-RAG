"""
Cross-Encoder Reranking Implementation

Provides production-grade reranking using cross-encoder models for
precise query-document relevance scoring.
"""

from typing import List, Optional, Tuple
from langchain.schema import Document
from sentence_transformers import CrossEncoder
import numpy as np


class CrossEncoderReranker:
    """
    Cross-encoder based reranker for precision document ranking.

    Unlike bi-encoders that encode query and document separately, cross-encoders
    process the concatenated [query, document] pair with full attention, achieving
    significantly higher accuracy at the cost of inference speed.

    Performance Characteristics:
        - Speed: ~10-50 documents/second (GPU) vs 1000s/sec for bi-encoders
        - Accuracy: +15-25% over bi-encoder-only ranking
        - Latency: +200-500ms for reranking 50 candidates
        - Use case: Final reranking of top-K candidates (K=20-100)

    Attributes:
        model: Loaded cross-encoder model
        model_name: Name of the cross-encoder model
    """

    # Popular cross-encoder models for different use cases
    MODELS = {
        "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Fast, good for general domains
        "accurate": "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Slower, higher accuracy
        "qa": "cross-encoder/qnli-distilroberta-base",  # Optimized for Q&A
        "multilingual": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",  # 100+ languages
    }

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name or shortcut from MODELS dict
            device: Device to run model on ('cuda', 'cpu', or None for auto)
            max_length: Maximum token length for input pairs

        Example:
            >>> reranker = CrossEncoderReranker("fast")
            >>> reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-12-v2")
        """
        # Resolve model shortcuts
        if model_name in self.MODELS:
            model_name = self.MODELS[model_name]

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
        return_scores: bool = True,
    ) -> List[Document]:
        """
        Rerank documents using cross-encoder scores.

        Args:
            query: User query string
            documents: List of candidate documents (typically 20-100)
            top_k: Number of top documents to return
            return_scores: If True, add 'rerank_score' to document metadata

        Returns:
            List of top-K reranked documents (highest scoring first)

        Example:
            >>> docs = retriever.get_relevant_documents(query, top_k=50)
            >>> reranked = reranker.rerank(query, docs, top_k=5)
        """
        if not documents:
            return []

        # Create [query, document] pairs
        pairs = [[query, doc.page_content] for doc in documents]

        # Score all pairs (this is the expensive operation)
        scores = self.model.predict(pairs)

        # Combine documents with scores
        scored_docs = list(zip(documents, scores))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Extract top K documents
        reranked = []
        for doc, score in scored_docs[:top_k]:
            if return_scores:
                # Add score to metadata for debugging/monitoring
                doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        return reranked

    def rerank_with_threshold(
        self,
        query: str,
        documents: List[Document],
        score_threshold: float = 0.5,
        max_results: int = 10,
    ) -> List[Document]:
        """
        Rerank and filter documents by minimum score threshold.

        Useful for ensuring quality when you'd rather return fewer results
        than include low-relevance documents.

        Args:
            query: User query string
            documents: List of candidate documents
            score_threshold: Minimum score to include document (0.0 to 1.0)
            max_results: Maximum number of results to return

        Returns:
            Documents above threshold, sorted by score
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        # Filter and sort
        filtered = [
            (doc, score)
            for doc, score in zip(documents, scores)
            if score >= score_threshold
        ]
        filtered.sort(key=lambda x: x[1], reverse=True)

        # Return top results
        reranked = []
        for doc, score in filtered[:max_results]:
            doc.metadata["rerank_score"] = float(score)
            reranked.append(doc)

        return reranked

    def score_pairs(
        self, query: str, documents: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Score query-document pairs without reranking.

        Useful for analysis, debugging, or custom ranking logic.

        Args:
            query: User query string
            documents: List of documents to score

        Returns:
            List of (document, score) tuples in original order
        """
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        return list(zip(documents, scores))

    def get_score_statistics(
        self, query: str, documents: List[Document]
    ) -> dict:
        """
        Compute score statistics for analysis.

        Args:
            query: User query string
            documents: List of documents to analyze

        Returns:
            Dictionary with score statistics
        """
        if not documents:
            return {
                "mean": 0.0,
                "median": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.model.predict(pairs)

        return {
            "mean": float(np.mean(scores)),
            "median": float(np.median(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "count": len(scores),
        }


class RerankedRetriever:
    """
    Wrapper that combines any base retriever with cross-encoder reranking.

    This is the production-ready component that integrates hybrid retrieval
    with reranking and optional context optimization.

    Attributes:
        base_retriever: Any LangChain-compatible retriever
        reranker: CrossEncoderReranker instance
        top_k: Number of final documents to return
        apply_context_reordering: Whether to apply "Lost in the Middle" fix
    """

    def __init__(
        self,
        base_retriever,
        reranker: Optional[CrossEncoderReranker] = None,
        top_k: int = 5,
        apply_context_reordering: bool = True,
    ):
        """
        Initialize reranked retriever.

        Args:
            base_retriever: Base retriever (e.g., HybridRetriever, EnsembleRetriever)
            reranker: CrossEncoderReranker instance (creates default if None)
            top_k: Number of documents to return after reranking
            apply_context_reordering: Apply "Lost in the Middle" optimization
        """
        self.base_retriever = base_retriever
        self.reranker = reranker or CrossEncoderReranker("fast")
        self.top_k = top_k
        self.apply_context_reordering = apply_context_reordering

    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Retrieve and rerank documents.

        Pipeline:
            1. Get candidates from base retriever (50-100 docs)
            2. Rerank with cross-encoder (top K)
            3. Optionally apply context reordering

        Args:
            query: User query string

        Returns:
            List of top-K reranked (and optionally reordered) documents
        """
        # Step 1: Get candidates from base retriever
        candidates = self.base_retriever.get_relevant_documents(query)

        # Step 2: Rerank with cross-encoder
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k)

        # Step 3: Apply context reordering if enabled
        if self.apply_context_reordering:
            from .utils import reorder_documents_lost_in_middle

            reranked = reorder_documents_lost_in_middle(reranked)

        return reranked

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Note: Cross-encoder reranking is still synchronous (CPU/GPU bound).
        Async is only beneficial for the initial retrieval step.

        Args:
            query: User query string

        Returns:
            List of top-K reranked documents
        """
        # Step 1: Async retrieval
        if hasattr(self.base_retriever, "aget_relevant_documents"):
            candidates = await self.base_retriever.aget_relevant_documents(query)
        else:
            candidates = self.base_retriever.get_relevant_documents(query)

        # Step 2 & 3: Reranking and reordering (sync)
        reranked = self.reranker.rerank(query, candidates, top_k=self.top_k)

        if self.apply_context_reordering:
            from .utils import reorder_documents_lost_in_middle

            reranked = reorder_documents_lost_in_middle(reranked)

        return reranked


if __name__ == "__main__":
    # Example usage
    from .retrieval import HybridRetriever, create_sample_documents

    # Setup
    docs = create_sample_documents()
    hybrid_retriever = HybridRetriever(docs)

    # Test reranking
    print("=== Testing Cross-Encoder Reranking ===")
    reranker = CrossEncoderReranker("fast")

    query = "How do I fix authentication timeouts?"
    candidates = hybrid_retriever.get_relevant_documents(query, top_k=6)

    print(f"\nQuery: {query}")
    print(f"Candidates from hybrid search: {len(candidates)}")

    reranked = reranker.rerank(query, candidates, top_k=3)

    print("\nTop 3 after reranking:")
    for i, doc in enumerate(reranked, 1):
        score = doc.metadata.get("rerank_score", 0)
        print(f"{i}. [Score: {score:.3f}] {doc.metadata['doc_id']}")
        print(f"   {doc.page_content[:100]}...\n")
