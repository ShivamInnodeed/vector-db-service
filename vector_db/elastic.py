"""
Elasticsearch client and index management for VectorDB service.

Handles Elasticsearch HTTP client initialization, index creation with dense_vector mapping,
and KNN search operations.
"""

import logging
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch
from elasticsearch.exceptions import RequestError

from vector_db.config import VectorDBConfig

logger = logging.getLogger(__name__)


class ElasticsearchClient:
    """Elasticsearch client wrapper for vector operations."""
    
    def __init__(self, config: VectorDBConfig):
        """
        Initialize Elasticsearch client.
        
        Args:
            config: VectorDB configuration
        
        Raises:
            ConnectionError: If Elasticsearch is not reachable
        """
        self.config = config
        self.client = Elasticsearch(
            [config.elasticsearch_url],
            timeout=config.timeout,
            max_retries=config.max_retries,
            retry_on_timeout=True,
        )
        
        # Validate connection on initialization
        try:
            if not self.client.ping():
                raise ConnectionError(f"Cannot connect to Elasticsearch at {config.elasticsearch_url}")
        except Exception as e:
            logger.warning(f"Elasticsearch connection check failed: {e}. Will retry on first operation.")
    
    def ensure_index_exists(self, index_name: str, vector_dimension: Optional[int] = None) -> None:
        """
        Ensure index exists with proper dense_vector mapping.
        
        Uses a try-except pattern to handle race conditions when multiple
        clients create the same index concurrently.
        
        Args:
            index_name: Name of the index
            vector_dimension: Dimension of vectors (defaults to config.vector_dimension)
        """
        dimension = vector_dimension or self.config.vector_dimension
        
        # Validate index name (basic ES naming rules)
        if not index_name or not isinstance(index_name, str):
            raise ValueError("index_name must be a non-empty string")
        if not index_name[0].islower():
            raise ValueError("index_name must start with a lowercase letter")
        if any(c in index_name for c in ['\\', '/', '*', '?', '"', '<', '>', '|', ' ', ',', '#', ':']):
            raise ValueError(f"index_name contains invalid characters: {index_name}")
        
        if self.client.indices.exists(index=index_name):
            logger.debug(f"Index {index_name} already exists")
            return
        
        mapping = {
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dimension,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "text": {
                        "type": "text",
                    },
                    "metadata": {
                        "type": "object",
                        "enabled": True,
                    },
                }
            }
        }
        
        try:
            # ES 8.x: use direct parameters instead of deprecated body=
            self.client.indices.create(index=index_name, mappings=mapping["mappings"])
            logger.info(f"Created index {index_name} with dense_vector mapping (dimension={dimension})")
        except RequestError as e:
            if "resource_already_exists_exception" in str(e):
                logger.debug(f"Index {index_name} was created concurrently")
            else:
                raise
    
    def knn_search(
        self,
        index_name: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform pure vector KNN search on Elasticsearch index (semantic search).
        
        Args:
            index_name: Name of the index to search
            query_vector: Query embedding vector
            top_k: Number of results to return
            filters: Optional filters to apply (e.g., {"metadata.status": "active"})
        
        Returns:
            List of Elasticsearch hits
        """
        knn_query: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 10,  # Increase candidates for better recall
        }
        
        # Add filters if provided
        if filters:
            knn_query["filter"] = self._build_filter_query(filters)
        
        try:
            # Elasticsearch 9.x: knn query is expressed in the body
            response = self.client.search(
                index=index_name,
                body={
                    "knn": knn_query,
                    "size": top_k,
                    "_source": ["text", "metadata"],
                },
            )
            hits = response.get("hits", {}).get("hits", [])
            logger.debug(f"KNN search returned {len(hits)} results")
            return hits
        except Exception as e:
            logger.error(f"KNN search failed: {e}")
            raise

    def bm25_search(
        self,
        index_name: str,
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform BM25 keyword search on Elasticsearch index.

        Args:
            index_name: Name of the index to search
            query_text: Raw text query for keyword matching
            top_k: Number of results to return
            filters: Optional filters to apply (e.g., {"metadata.status": "active"})

        Returns:
            List of Elasticsearch hits with BM25 _score
        """
        # IMPORTANT:
        # Do NOT use "metadata.*" here because some metadata fields are numeric
        # (e.g. chunk_id, cosine_score). multi_match over numeric fields with
        # a string query like "credit card apply" causes ES to throw
        # "failed to create query: For input string: \"...\"" errors.
        # Restrict BM25 to known text fields only.
        bm25_query: Dict[str, Any] = {
            "multi_match": {
                "query": query_text,
                "fields": [
                    "text^2",             # main content
                    "metadata.link^3",    # URL often very informative
                    "metadata.header^2",  # section / title
                    "metadata.section",   # optional section name
                ],
            }
        }
        if filters:
            bm25_query = {
                "bool": {
                    "must": [bm25_query],
                    "filter": [self._build_filter_query(filters)],
                }
            }
        try:
            response = self.client.search(
                index=index_name,
                body={
                    "query": bm25_query,
                    "size": top_k,
                    "_source": ["text", "metadata"],
                },
            )
            hits = response.get("hits", {}).get("hits", [])
            logger.debug(f"BM25 search returned {len(hits)} results")
            return hits
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            raise

    def hybrid_search(
        self,
        index_name: str,
        query_vector: List[float],
        query_text: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search: semantic KNN + BM25 keyword search.
        
        Uses Elasticsearch 9.x hybrid search support by combining a knn clause
        with a BM25 query and Reciprocal Rank Fusion (RRF) ranking.
        
        Args:
            index_name: Name of the index to search
            query_vector: Query embedding vector (for semantic search)
            query_text: Raw query text (for BM25 keyword search)
            top_k: Number of results to return
            filters: Optional filters to apply (metadata constraints)
        
        Returns:
            List of Elasticsearch hits ranked by hybrid score
        """
        knn_query: Dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": top_k,
            "num_candidates": top_k * 10,
        }
        if filters:
            knn_query["filter"] = self._build_filter_query(filters)

        # BM25 keyword query over text (and optionally metadata)
        bm25_query: Dict[str, Any] = {
            "multi_match": {
                "query": query_text,
                "fields": ["text^2", "metadata.*"],
            }
        }

        # Hybrid search with Reciprocal Rank Fusion (RRF)
        body: Dict[str, Any] = {
            "knn": knn_query,
            "query": bm25_query,
            "rank": {
                "rrf": {
                    "window_size": top_k * 10,
                    "rank_constant": 60,
                }
            },
            "size": top_k,
            "_source": ["text", "metadata"],
        }

        try:
            response = self.client.search(index=index_name, body=body)
            hits = response.get("hits", {}).get("hits", [])
            logger.debug(f"Hybrid search returned {len(hits)} results")
            return hits
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise
    
    def _build_filter_query(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Elasticsearch filter query from filters dict.
        
        Supports term queries for exact matches. For complex filters (range, exists, etc.),
        pass pre-built Elasticsearch query dicts directly.
        
        Args:
            filters: Dictionary of field:value pairs to filter on, or pre-built ES query dict
        
        Returns:
            Elasticsearch query dict
        """
        # If filters is already a query dict (e.g., {"bool": {...}}), return as-is
        if isinstance(filters, dict) and any(key in filters for key in ["bool", "term", "range", "exists", "match"]):
            return filters
        
        if len(filters) == 1:
            field, value = next(iter(filters.items()))
            return {"term": {field: value}}
        
        # Multiple filters - use bool must
        must_clauses = [
            {"term": {field: value}}
            for field, value in filters.items()
        ]
        return {"bool": {"must": must_clauses}}
    
    def bulk_index(
        self,
        index_name: str,
        documents: List[Dict[str, Any]],
        refresh: bool = False,
        chunk_size: int = 500,
    ) -> tuple[int, int]:
        """
        Bulk index documents into Elasticsearch.
        
        Args:
            index_name: Name of the index
            documents: List of documents to index (each should have _id and _source)
            refresh: Whether to refresh the index after indexing (default: False)
            chunk_size: Number of documents per bulk request (default: 500)
        
        Returns:
            Tuple of (success_count, failed_count)
        """
        from elasticsearch.helpers import bulk, streaming_bulk
        
        actions = [
            {
                "_index": index_name,
                "_id": doc.get("_id") or doc.get("id"),
                "_source": doc.get("_source") or doc,
            }
            for doc in documents
        ]
        
        # Process in chunks for large batches
        success_count = 0
        failed_count = 0
        
        try:
            for chunk_start in range(0, len(actions), chunk_size):
                chunk = actions[chunk_start:chunk_start + chunk_size]
                success, failed = bulk(self.client, chunk, raise_on_error=False, refresh=refresh)
                success_count += success
                if failed:
                    failed_count += len(failed)
                    logger.warning(f"Failed to index {len(failed)} documents in chunk")
                    for item in failed:
                        logger.debug(f"Failed item: {item}")
            
            logger.info(f"Bulk indexed {success_count} documents into {index_name} ({failed_count} failed)")
            return (success_count, failed_count)
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """
        Check Elasticsearch cluster health.
        
        Returns:
            True if cluster is healthy (green or yellow), False otherwise
        """
        try:
            health = self.client.cluster.health()
            status = health.get("status")
            is_healthy = status in ["green", "yellow"]
            if status == "red":
                logger.warning("Elasticsearch cluster status is RED - data loss risk")
            return is_healthy
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
