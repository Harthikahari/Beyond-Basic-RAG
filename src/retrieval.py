"""
Hybrid Retrieval Implementation

Combines BM25 sparse retrieval with dense vector search using
Reciprocal Rank Fusion (RRF) for optimal keyword and semantic matching.
"""

from typing import List, Optional
from langchain.schema import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


class HybridRetriever:
    """
    Production-grade hybrid retriever combining BM25 and dense vector search.

    This retriever addresses the "Lexical Gap" by running both keyword-based
    (BM25) and semantic (dense vector) search in parallel, then merging results
    using Reciprocal Rank Fusion.

    Attributes:
        bm25_retriever: Sparse keyword-based retriever
        dense_retriever: Dense vector-based semantic retriever
        ensemble_retriever: Combined retriever using RRF
        bm25_weight: Weight for BM25 scores (default: 0.5)
        dense_weight: Weight for dense retriever scores (default: 0.5)
    """

    def __init__(
        self,
        documents: List[Document],
        embedding_model: str = "text-embedding-3-small",
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
        bm25_k: int = 25,
        dense_k: int = 25,
        rrf_c: int = 60,
    ):
        """
        Initialize hybrid retriever with document corpus.

        Args:
            documents: List of documents to index
            embedding_model: OpenAI embedding model name
            bm25_weight: Weight for BM25 retriever (0.0 to 1.0)
            dense_weight: Weight for dense retriever (0.0 to 1.0)
            bm25_k: Number of documents to retrieve with BM25
            dense_k: Number of documents to retrieve with dense search
            rrf_c: RRF constant (typically 60)

        Weight Guidelines:
            - High-precision domains (legal, medical): bm25=0.6, dense=0.4
            - Conversational queries: bm25=0.3, dense=0.7
            - Mixed workloads: bm25=0.5, dense=0.5
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # Initialize BM25 (sparse) retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = bm25_k

        # Initialize dense vector retriever
        embeddings = OpenAIEmbeddings(model=embedding_model)
        vectorstore = Chroma.from_documents(documents, embeddings)
        self.dense_retriever = vectorstore.as_retriever(
            search_kwargs={"k": dense_k}
        )

        # Create ensemble retriever with RRF
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[bm25_weight, dense_weight],
            c=rrf_c,
        )

    def get_relevant_documents(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents using hybrid search.

        Args:
            query: User query string
            top_k: Number of documents to return (None = all merged results)

        Returns:
            List of documents ranked by RRF score
        """
        results = self.ensemble_retriever.get_relevant_documents(query)

        if top_k is not None:
            results = results[:top_k]

        return results

    async def aget_relevant_documents(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Async version of get_relevant_documents.

        Args:
            query: User query string
            top_k: Number of documents to return

        Returns:
            List of documents ranked by RRF score
        """
        results = await self.ensemble_retriever.aget_relevant_documents(query)

        if top_k is not None:
            results = results[:top_k]

        return results

    def update_weights(self, bm25_weight: float, dense_weight: float) -> None:
        """
        Update retriever weights dynamically.

        Useful for A/B testing or query-specific weight optimization.

        Args:
            bm25_weight: New weight for BM25 retriever
            dense_weight: New weight for dense retriever
        """
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

        # Recreate ensemble with new weights
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[bm25_weight, dense_weight],
            c=60,
        )


def create_sample_documents() -> List[Document]:
    """
    Create sample document corpus for testing.

    Returns:
        List of sample documents covering various query types
    """
    return [
        Document(
            page_content="Error 503-AUTH-TIMEOUT occurs when authentication takes longer than 30 seconds. "
            "This typically happens during peak load or when the authentication service is degraded.",
            metadata={"doc_id": "ERR-001", "category": "errors", "severity": "high"},
        ),
        Document(
            page_content="Error 504-GATEWAY-TIMEOUT indicates that an upstream service did not respond "
            "within the configured timeout period. Check service health and network connectivity.",
            metadata={"doc_id": "ERR-002", "category": "errors", "severity": "high"},
        ),
        Document(
            page_content="Product SKU-2847-B is an enterprise-grade widget with 99.9% SLA guarantee. "
            "It includes 24/7 support and advanced monitoring capabilities.",
            metadata={"doc_id": "PROD-847", "category": "products", "tier": "enterprise"},
        ),
        Document(
            page_content="Authentication timeouts can be extended via the config.auth.timeout setting. "
            "Default value is 30 seconds. Maximum recommended value is 120 seconds.",
            metadata={"doc_id": "CFG-012", "category": "configuration", "component": "auth"},
        ),
        Document(
            page_content="Session expiration is controlled by the session.max_age parameter. "
            "When a session expires, users must re-authenticate.",
            metadata={"doc_id": "CFG-045", "category": "configuration", "component": "session"},
        ),
        Document(
            page_content="Login delays can be caused by rate limiting, database contention, "
            "or network latency. Enable debug logging for detailed diagnostics.",
            metadata={"doc_id": "TRB-023", "category": "troubleshooting", "component": "auth"},
        ),
    ]


if __name__ == "__main__":
    # Example usage
    docs = create_sample_documents()
    retriever = HybridRetriever(docs)

    # Test exact keyword match
    print("=== Test 1: Exact Keyword Match ===")
    results = retriever.get_relevant_documents("503-AUTH-TIMEOUT", top_k=3)
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata['doc_id']}: {doc.page_content[:80]}...")

    # Test semantic query
    print("\n=== Test 2: Semantic Query ===")
    results = retriever.get_relevant_documents(
        "How do I fix authentication timeouts?", top_k=3
    )
    for i, doc in enumerate(results, 1):
        print(f"{i}. {doc.metadata['doc_id']}: {doc.page_content[:80]}...")
