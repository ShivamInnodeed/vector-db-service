"""
Main VectorDB client for semantic search and document indexing.

Provides the primary interface for Langgraph Node 2 to perform semantic search
and for Knowledge Ingestion flow to index documents.
"""

import logging
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from vector_db.config import HybridSearchConfig, VectorDBConfig
from vector_db.elastic import ElasticsearchClient
from vector_db.models import Document, SearchResult
from vector_db.search import (
    BM25SearchStrategy,
    HybridSearchStrategy,
    KNNSearchStrategy,
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class VectorDBClient:
    """Main client for VectorDB operations."""
    
    def __init__(self, config: Optional[VectorDBConfig] = None):
        """
        Initialize VectorDB client.
        
        Args:
            config: VectorDB configuration (defaults to default_config)
        """
        self.config = config or VectorDBConfig()
        self.es_client = ElasticsearchClient(self.config)
        self._knn_strategy = KNNSearchStrategy()
        self._bm25_strategy = BM25SearchStrategy()
        self._hybrid_strategy = HybridSearchStrategy(
            hybrid_config=getattr(self.config, "default_hybrid_config", None)
            or HybridSearchConfig(0.5, 0.5)
        )
        # Initialize OpenTelemetry if enabled
        if self.config.enable_telemetry:
            self._init_telemetry()
    
    def _init_telemetry(self) -> None:
        """Initialize OpenTelemetry instrumentation."""
        try:
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            
            # Only set tracer provider if not already set (avoid overwriting)
            current_provider = trace.get_tracer_provider()
            if isinstance(current_provider, TracerProvider):
                # Provider already configured, just use it
                logger.debug("OpenTelemetry tracer provider already configured")
                return
            
            # Set up tracer provider
            provider = TracerProvider()
            trace.set_tracer_provider(provider)
            
            # Add OTLP exporter if endpoint is configured
            if self.config.otlp_endpoint:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                otlp_exporter = OTLPSpanExporter(endpoint=self.config.otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                logger.info(f"OpenTelemetry OTLP exporter configured: {self.config.otlp_endpoint}")
            else:
                logger.debug("OpenTelemetry enabled but no OTLP endpoint configured")
        except ImportError:
            logger.warning("OpenTelemetry packages not available, telemetry disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize telemetry: {e}")
    
    def search(
        self,
        query_embedding: List[float],
        index: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform semantic search on Elasticsearch index.
        
        This is the primary method used by Langgraph Node 2 to find similar queries
        and retrieve answers.
        
        Args:
            query_embedding: Query vector embedding (list of floats)
            index: Name of the Elasticsearch index to search
            top_k: Number of results to return (default: 10)
            filters: Optional filters to apply (e.g., {"metadata.status": "active"})
        
        Returns:
            List of SearchResult objects sorted by relevance score
        
        Example:
            >>> client = VectorDBClient()
            >>> results = client.search(
            ...     query_embedding=[0.1, 0.2, 0.3, ...],
            ...     index="livechat_answers",
            ...     top_k=5
            ... )
            >>> for result in results:
            ...     print(f"{result.id}: {result.score} - {result.text}")
        """
        span = tracer.start_span("vectordb.search")
        span.set_attribute("index", index)
        span.set_attribute("top_k", top_k)
        span.set_attribute("has_filters", filters is not None)
        
        try:
            # Validate input
            if not query_embedding:
                raise ValueError("query_embedding cannot be empty")
            if not isinstance(query_embedding, list):
                raise TypeError("query_embedding must be a list")
            if not all(isinstance(x, (int, float)) for x in query_embedding):
                raise TypeError("query_embedding must contain only numbers")
            if not index:
                raise ValueError("index name cannot be empty")
            if not isinstance(index, str):
                raise TypeError("index must be a string")
            if not isinstance(top_k, int):
                raise TypeError("top_k must be an integer")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            
            # Ensure index exists
            self.es_client.ensure_index_exists(index, len(query_embedding))
            
            # Delegate to KNN strategy
            results = self._knn_strategy.execute(
                self.es_client,
                index=index,
                top_k=top_k,
                filters=filters,
                query_embedding=query_embedding,
            )
            
            span.set_attribute("results_count", len(results))
            span.set_status(Status(StatusCode.OK))
            logger.info(f"Search completed: {len(results)} results from index {index}")
            
            return results
        
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            logger.error(f"Search failed: {e}")
            raise
        
        finally:
            span.end()

    def search_bm25(
        self,
        query_text: str,
        index: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Perform BM25 keyword search on the index.
        
        Args:
            query_text: Raw text query for keyword matching
            index: Name of the Elasticsearch index to search
            top_k: Number of results to return (default: 10)
            filters: Optional filters to apply (e.g., {"metadata.status": "active"})
        
        Returns:
            List of SearchResult objects sorted by BM25 score
        """
        span = tracer.start_span("vectordb.search_bm25")
        span.set_attribute("index", index)
        span.set_attribute("top_k", top_k)
        try:
            if not query_text or not isinstance(query_text, str) or not query_text.strip():
                raise ValueError("query_text must be a non-empty string")
            if not index:
                raise ValueError("index name cannot be empty")
            if not isinstance(top_k, int) or top_k <= 0:
                raise ValueError("top_k must be a positive integer")
            self.es_client.ensure_index_exists(index)
            results = self._bm25_strategy.execute(
                self.es_client,
                index=index,
                top_k=top_k,
                filters=filters,
                query_text=query_text,
            )
            span.set_attribute("results_count", len(results))
            span.set_status(Status(StatusCode.OK))
            logger.info(f"BM25 search completed: {len(results)} results from index {index}")
            return results
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            logger.error(f"BM25 search failed: {e}")
            raise
        finally:
            span.end()

    def search_hybrid(
        self,
        query_embedding: List[float],
        query_text: str,
        index: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        *,
        knn_weight: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        hybrid_config: Optional[HybridSearchConfig] = None,
    ) -> List[SearchResult]:
        """
        Perform hybrid search (semantic KNN + BM25 keyword search) on Elasticsearch index.
        
        When knn_weight and bm25_weight (or hybrid_config) are provided, uses weighted
        score combination with normalization:
            final_score = knn_weight * normalized_knn_score + bm25_weight * normalized_bm25_score
        Otherwise uses Elasticsearch RRF ranking for backward compatibility.
        
        Args:
            query_embedding: Query vector embedding (list of floats) for semantic search
            query_text: Raw query text for BM25 keyword search
            index: Name of the Elasticsearch index to search
            top_k: Number of results to return (default: 10)
            filters: Optional filters to apply (e.g., {"metadata.status": "active"})
            knn_weight: Weight for KNN score in [0, 1]; must sum with bm25_weight to 1.0
            bm25_weight: Weight for BM25 score in [0, 1]; must sum with knn_weight to 1.0
            hybrid_config: Optional config for weights (overridden by knn_weight/bm25_weight if set)
        
        Returns:
            List of SearchResult objects sorted by hybrid relevance score
        """
        span = tracer.start_span("vectordb.search_hybrid")
        span.set_attribute("index", index)
        span.set_attribute("top_k", top_k)
        span.set_attribute("has_filters", filters is not None)
        
        try:
            # Validate inputs
            if not query_embedding:
                raise ValueError("query_embedding cannot be empty")
            if not isinstance(query_embedding, list):
                raise TypeError("query_embedding must be a list")
            if not all(isinstance(x, (int, float)) for x in query_embedding):
                raise TypeError("query_embedding must contain only numbers")
            if not query_text or not isinstance(query_text, str) or not query_text.strip():
                raise ValueError("query_text must be a non-empty string")
            if not index:
                raise ValueError("index name cannot be empty")
            if not isinstance(index, str):
                raise TypeError("index must be a string")
            if not isinstance(top_k, int):
                raise TypeError("top_k must be an integer")
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            
            # Ensure index exists
            self.es_client.ensure_index_exists(index, len(query_embedding))
            
            use_weighted = (
                (knn_weight is not None and bm25_weight is not None) or hybrid_config is not None
            )
            if use_weighted:
                if knn_weight is not None and bm25_weight is not None:
                    from vector_db.config import _validate_weights
                    _validate_weights(knn_weight, bm25_weight)
                strategy = HybridSearchStrategy(
                    hybrid_config=hybrid_config or getattr(
                        self.config, "default_hybrid_config", None
                    ) or HybridSearchConfig(0.5, 0.5)
                )
                results = strategy.execute(
                    self.es_client,
                    index=index,
                    top_k=top_k,
                    filters=filters,
                    query_embedding=query_embedding,
                    query_text=query_text,
                    knn_weight=knn_weight,
                    bm25_weight=bm25_weight,
                )
            else:
                hits = self.es_client.hybrid_search(
                    index_name=index,
                    query_vector=query_embedding,
                    query_text=query_text,
                    top_k=top_k,
                    filters=filters,
                )
                results = [SearchResult.from_es_hit(hit) for hit in hits]
            
            span.set_attribute("results_count", len(results))
            span.set_status(Status(StatusCode.OK))
            logger.info(f"Hybrid search completed: {len(results)} results from index {index}")
            
            return results
        
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            logger.error(f"Hybrid search failed: {e}")
            raise
        
        finally:
            span.end()
    
    def index_documents(
        self,
        index: str,
        documents: List[Document],
        vector_dimension: Optional[int] = None,
        refresh: bool = False,
    ) -> int:
        """
        Index documents into Elasticsearch with their embeddings.
        
        This method is used by the Knowledge Ingestion flow (Node 6: Indexing & Storing)
        to store processed and embedded documents.
        
        Args:
            index: Name of the Elasticsearch index
            documents: List of Document objects to index
            vector_dimension: Dimension of vectors (defaults to config.vector_dimension or inferred from first doc)
            refresh: Whether to refresh the index after indexing (default: False)
        
        Returns:
            Number of documents successfully indexed
        
        Example:
            >>> client = VectorDBClient()
            >>> docs = [
            ...     Document(
            ...         id="doc1",
            ...         embedding=[0.1, 0.2, ...],
            ...         text="Answer text here",
            ...         metadata={"source": "kb", "category": "faq"}
            ...     )
            ... ]
            >>> count = client.index_documents("livechat_answers", docs)
        """
        span = tracer.start_span("vectordb.index_documents")
        span.set_attribute("index", index)
        span.set_attribute("documents_count", len(documents))
        
        try:
            if not documents:
                raise ValueError("documents list cannot be empty")
            if not index:
                raise ValueError("index name cannot be empty")
            
            # Determine vector dimension
            dim = vector_dimension or self.config.vector_dimension
            if documents and documents[0].embedding:
                dim = len(documents[0].embedding)
            
            # Ensure index exists with correct mapping
            self.es_client.ensure_index_exists(index, dim)
            
            # Prepare documents for bulk indexing
            es_docs = []
            for doc in documents:
                if len(doc.embedding) != dim:
                    raise ValueError(
                        f"Document {doc.id} has embedding dimension {len(doc.embedding)}, "
                        f"expected {dim}"
                    )
                es_docs.append({
                    "_id": doc.id,
                    "_source": doc.to_dict(),
                })
            
            # Bulk index
            success_count, failed_count = self.es_client.bulk_index(index, es_docs, refresh=refresh)
            
            span.set_attribute("indexed_count", success_count)
            span.set_attribute("failed_count", failed_count)
            span.set_status(Status(StatusCode.OK))
            logger.info(f"Indexed {success_count} documents into {index} ({failed_count} failed)")
            
            if failed_count > 0:
                logger.warning(f"{failed_count} documents failed to index")
            
            return success_count
        
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            logger.error(f"Indexing failed: {e}")
            raise
        
        finally:
            span.end()
    
    def health_check(self) -> bool:
        """
        Check health of Elasticsearch cluster.
        
        Returns:
            True if cluster is healthy, False otherwise
        """
        span = tracer.start_span("vectordb.health_check")
        try:
            is_healthy = self.es_client.health_check()
            span.set_attribute("healthy", is_healthy)
            span.set_status(Status(StatusCode.OK))
            return is_healthy
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            return False
        finally:
            span.end()
