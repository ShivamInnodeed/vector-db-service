"""
KNN (vector / semantic) search strategy.
"""

import logging
from typing import Any, Dict, List, Optional

from vector_db.models import SearchResult
from vector_db.search.base import SearchStrategy

logger = logging.getLogger(__name__)


class KNNSearchStrategy(SearchStrategy):
    """
    Vector-based KNN search using query embedding and cosine similarity.
    
    Returns ranked results with similarity scores from Elasticsearch
    dense_vector KNN.
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
        Perform KNN search with query embedding.
        
        Required in kwargs: query_embedding (List[float]).
        """
        query_embedding: Optional[List[float]] = kwargs.get("query_embedding")
        if not query_embedding:
            raise ValueError("KNN strategy requires query_embedding in kwargs")
        hits = es_client.knn_search(
            index_name=index,
            query_vector=query_embedding,
            top_k=top_k,
            filters=filters,
        )
        return [SearchResult.from_es_hit(h) for h in hits]
