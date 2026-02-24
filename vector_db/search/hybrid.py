"""
Hybrid search strategy: weighted combination of KNN and BM25 with score normalization.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from vector_db.config import HybridSearchConfig, _validate_weights
from vector_db.models import SearchResult
from vector_db.search.base import SearchStrategy

logger = logging.getLogger(__name__)


def _min_max_normalize(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1] using min-max normalization.
    
    If all scores are equal or only one value, returns 1.0 for all
    to avoid division by zero.
    """
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s <= min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def _merge_and_combine(
    knn_results: List[SearchResult],
    bm25_results: List[SearchResult],
    knn_weight: float,
    bm25_weight: float,
) -> List[SearchResult]:
    """
    Normalize KNN and BM25 scores, then combine by doc id with weighted sum.
    
    final_score = knn_weight * normalized_knn + bm25_weight * normalized_bm25.
    Documents appearing in only one list get 0 for the missing score.
    """
    knn_scores = [r.score for r in knn_results]
    bm25_scores = [r.score for r in bm25_results]
    norm_knn = _min_max_normalize(knn_scores)
    norm_bm25 = _min_max_normalize(bm25_scores)

    knn_by_id: Dict[str, Tuple[SearchResult, float]] = {
        r.id: (r, norm_knn[i]) for i, r in enumerate(knn_results)
    }
    bm25_by_id: Dict[str, Tuple[SearchResult, float]] = {
        r.id: (r, norm_bm25[i]) for i, r in enumerate(bm25_results)
    }
    all_ids = set(knn_by_id) | set(bm25_by_id)

    combined: List[SearchResult] = []
    for doc_id in all_ids:
        res_knn, nk = knn_by_id.get(doc_id, (None, 0.0))
        res_bm25, nb = bm25_by_id.get(doc_id, (None, 0.0))
        final_score = knn_weight * nk + bm25_weight * nb
        # Prefer result that has content (use either hit for id/text/metadata)
        base = res_knn or res_bm25
        if base is None:
            continue
        combined.append(
            SearchResult(
                id=base.id,
                score=final_score,
                text=base.text,
                metadata=base.metadata,
            )
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "hybrid_score doc_id=%s norm_knn=%.4f norm_bm25=%.4f knn_w=%.2f bm25_w=%.2f final=%.4f",
                doc_id,
                nk,
                nb,
                knn_weight,
                bm25_weight,
                final_score,
            )
    combined.sort(key=lambda r: r.score, reverse=True)
    return combined


class HybridSearchStrategy(SearchStrategy):
    """
    Hybrid search: run KNN and BM25, normalize scores, then combine with
    configurable weights:
        final_score = knn_weight * normalized_knn_score + bm25_weight * normalized_bm25_score
    """

    def __init__(self, hybrid_config: Optional[HybridSearchConfig] = None):
        """
        Initialize with optional hybrid weight config.
        
        Args:
            hybrid_config: Weights for KNN and BM25. Defaults to 0.5/0.5.
        """
        self.hybrid_config = hybrid_config or HybridSearchConfig(0.5, 0.5)

    def execute(
        self,
        es_client: Any,
        index: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[SearchResult]:
        """
        Run KNN and BM25, normalize scores, combine with configured weights, return top_k.
        
        Required in kwargs: query_embedding (List[float]), query_text (str).
        Optional: knn_weight, bm25_weight (override config); validated if provided.
        """
        query_embedding = kwargs.get("query_embedding")
        query_text = kwargs.get("query_text")
        if not query_embedding:
            raise ValueError("Hybrid strategy requires query_embedding in kwargs")
        if not query_text or not isinstance(query_text, str):
            raise ValueError("Hybrid strategy requires non-empty query_text in kwargs")

        knn_weight = kwargs.get("knn_weight", self.hybrid_config.knn_weight)
        bm25_weight = kwargs.get("bm25_weight", self.hybrid_config.bm25_weight)
        _validate_weights(knn_weight, bm25_weight)

        # Fetch more candidates so we have overlap for fusion; then take top_k
        fetch_k = max(top_k * 3, 50)
        knn_hits = es_client.knn_search(
            index_name=index,
            query_vector=query_embedding,
            top_k=fetch_k,
            filters=filters,
        )
        bm25_hits = es_client.bm25_search(
            index_name=index,
            query_text=query_text,
            top_k=fetch_k,
            filters=filters,
        )
        knn_results = [SearchResult.from_es_hit(h) for h in knn_hits]
        bm25_results = [SearchResult.from_es_hit(h) for h in bm25_hits]

        combined = _merge_and_combine(
            knn_results,
            bm25_results,
            knn_weight,
            bm25_weight,
        )
        return combined[:top_k]
