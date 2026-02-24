"""
VectorDB Service - Python module for semantic search and vector storage using Elasticsearch.

This module provides a Python library interface for vector database operations,
designed to be used by Langgraph Node 2 in the Livechat suggestion flow.

Usage:
    from vector_db import VectorDBClient
    
    client = VectorDBClient()
    results = client.search(query_embedding=[0.1, 0.2, ...], index="livechat_answers", top_k=5)
"""

from vector_db.client import VectorDBClient
from vector_db.models import Document, SearchResult

__all__ = ["VectorDBClient", "Document", "SearchResult"]
__version__ = "0.1.0"
