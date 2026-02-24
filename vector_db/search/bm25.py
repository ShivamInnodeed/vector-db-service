"""
BM25 (keyword) search strategy.
"""

import logging
from typing import Any, Dict, List, Optional

from vector_db.models import SearchResult
from vector_db.search.base import SearchStrategy

logger = logging.getLogger(__name__)


class BM25SearchStrategy(SearchStrategy):
    """
    Keyword-based BM25 search using raw text query.
    
    Returns ranked results with BM25 scores from Elasticsearch
    multi_match query.
    """

    def execute(
        self,
        es_client: Any,
        index: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """
        Perform BM25 search with query text.
        
        Required in kwargs: query_text (str).
        """
        query_text: Optional[str] = kwargs.get("query_text")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("BM25 strategy requires non-empty query_text in kwargs")
        hits = es_client.bm25_search(
            index_name=index,
            query_text=query_text,
            top_k=top_k,
            filters=filters,
        )
        return [SearchResult.from_es_hit(h) for h in hits]
