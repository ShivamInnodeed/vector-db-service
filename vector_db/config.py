"""
Configuration for VectorDB service.

Loads configuration from environment variables with sensible defaults.
"""

import os
from typing import Optional


def _validate_weights(knn_weight: float, bm25_weight: float) -> None:
    """
    Validate hybrid search weights.
    
    Args:
        knn_weight: Weight for KNN (semantic) score in [0, 1].
        bm25_weight: Weight for BM25 (keyword) score in [0, 1].
    
    Raises:
        ValueError: If weights are outside [0, 1] or do not sum to 1.0.
    """
    if not (0.0 <= knn_weight <= 1.0):
        raise ValueError(f"knn_weight must be in [0, 1], got {knn_weight}")
    if not (0.0 <= bm25_weight <= 1.0):
        raise ValueError(f"bm25_weight must be in [0, 1], got {bm25_weight}")
    total = knn_weight + bm25_weight
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"knn_weight + bm25_weight must sum to 1.0, got {knn_weight} + {bm25_weight} = {total}"
        )


class HybridSearchConfig:
    """
    Configuration for hybrid search score combination.
    
    Used when combining KNN and BM25 results with:
        final_score = knn_weight * normalized_knn_score + bm25_weight * normalized_bm25_score
    """
    
    def __init__(
        self,
        knn_weight: float = 0.5,
        bm25_weight: float = 0.5,
    ):
        """
        Initialize hybrid search weights.
        
        Args:
            knn_weight: Weight for normalized KNN score (default 0.5).
            bm25_weight: Weight for normalized BM25 score (default 0.5).
        
        Raises:
            ValueError: If weights are invalid (see _validate_weights).
        """
        _validate_weights(knn_weight, bm25_weight)
        self.knn_weight = knn_weight
        self.bm25_weight = bm25_weight


class VectorDBConfig:
    """Configuration for VectorDB service."""
    
    def __init__(
        self,
        elasticsearch_url: Optional[str] = None,
        elasticsearch_index_prefix: Optional[str] = None,
        vector_dimension: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        enable_telemetry: Optional[bool] = None,
        otlp_endpoint: Optional[str] = None,
    ):
        """
        Initialize configuration.
        
        Args:
            elasticsearch_url: Elasticsearch cluster URL (default: from ELASTICSEARCH_URL env var or "http://localhost:9200")
            elasticsearch_index_prefix: Prefix for index names (default: from ELASTICSEARCH_INDEX_PREFIX or "vectordb")
            vector_dimension: Dimension of vector embeddings (default: from VECTOR_DIMENSION env var or 384)
            timeout: Request timeout in seconds (default: from ELASTICSEARCH_TIMEOUT env var or 30)
            max_retries: Maximum number of retries (default: from ELASTICSEARCH_MAX_RETRIES env var or 3)
            enable_telemetry: Enable OpenTelemetry instrumentation (default: from ENABLE_TELEMETRY env var or True)
            otlp_endpoint: OTLP endpoint for telemetry export (default: from OTLP_ENDPOINT env var or None)
        """
        self.elasticsearch_url = (
            elasticsearch_url
            or os.getenv("ELASTICSEARCH_URL")
        )
        self.elasticsearch_index_prefix = (
            elasticsearch_index_prefix
            or os.getenv("ELASTICSEARCH_INDEX_PREFIX", "vectordb")
        )
        self.vector_dimension = int(
            vector_dimension
            or os.getenv("VECTOR_DIMENSION", "384")
        )
        self.timeout = int(
            timeout
            or os.getenv("ELASTICSEARCH_TIMEOUT", "30")
        )
        self.max_retries = int(
            max_retries
            or os.getenv("ELASTICSEARCH_MAX_RETRIES", "3")
        )
        self.enable_telemetry = (
            enable_telemetry
            if enable_telemetry is not None
            else os.getenv("ENABLE_TELEMETRY", "true").lower() == "true"
        )
        self.otlp_endpoint = (
            otlp_endpoint
            or os.getenv("OTLP_ENDPOINT")
        )
        # Default hybrid search weights (can be overridden per call)
        self.default_hybrid_config = HybridSearchConfig(knn_weight=0.5, bm25_weight=0.5)


# Default configuration instance
default_config = VectorDBConfig()
