"""
Abstract base for search strategies (Strategy Pattern).

Enables pluggable KNN, BM25, and Hybrid search implementations
with a common interface for the VectorDB client.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from vector_db.models import SearchResult

logger = logging.getLogger(__name__)


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.
    
    Implementations (KNN, BM25, Hybrid) perform the actual search
    and return a list of SearchResult. The client delegates to
    the chosen strategy without coupling to concrete search logic.
    """

    @abstractmethod
    def execute(
        self,
        es_client: Any,
        index: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """
        Execute the search and return ranked results.
        
        Args:
            es_client: Elasticsearch client (ElasticsearchClient instance).
            index: Index name to search.
            top_k: Maximum number of results to return.
            filters: Optional metadata filters.
            **kwargs: Strategy-specific arguments (e.g. query_embedding, query_text).
        
        Returns:
            List of SearchResult sorted by relevance score.
        """
        pass
