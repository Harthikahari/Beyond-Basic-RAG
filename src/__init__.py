"""
Beyond Basic RAG: Hybrid Search and Reranking Implementation

This package provides production-ready retrieval augmented generation (RAG)
components with hybrid search and cross-encoder reranking capabilities.
"""

__version__ = "0.1.0"
__author__ = "Principal AI Architect"

from .retrieval import HybridRetriever
from .reranker import CrossEncoderReranker
from .utils import reorder_documents_lost_in_middle

__all__ = [
    "HybridRetriever",
    "CrossEncoderReranker",
    "reorder_documents_lost_in_middle",
]
