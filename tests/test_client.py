"""
Unit tests for VectorDBClient with mocked Elasticsearch.

These tests use mocks to avoid requiring a real Elasticsearch instance.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List

from vector_db import VectorDBClient, Document, SearchResult
from vector_db.config import HybridSearchConfig, VectorDBConfig


class TestVectorDBClient:
    """Test cases for VectorDBClient."""
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return VectorDBConfig(
            elasticsearch_url="http://localhost:9200",
            vector_dimension=384,
            enable_telemetry=False,
        )
    
    @pytest.fixture
    def client(self, mock_config):
        """Create a VectorDBClient instance with mocked ES."""
        with patch('vector_db.client.ElasticsearchClient'):
            return VectorDBClient(config=mock_config)
    
    @pytest.fixture
    def sample_embedding(self):
        """Sample query embedding."""
        return [0.1] * 384
    
    @pytest.fixture
    def sample_es_hits(self):
        """Sample Elasticsearch hits response."""
        return [
            {
                "_id": "doc1",
                "_score": 0.95,
                "_source": {
                    "text": "This is answer 1",
                    "metadata": {"category": "faq"}
                }
            },
            {
                "_id": "doc2",
                "_score": 0.87,
                "_source": {
                    "text": "This is answer 2",
                    "metadata": {"category": "troubleshooting"}
                }
            },
        ]
    
    def test_search_success(self, client, sample_embedding, sample_es_hits):
        """Test successful search operation."""
        # Mock ES client methods
        client.es_client.ensure_index_exists = Mock()
        client.es_client.knn_search = Mock(return_value=sample_es_hits)
        
        # Perform search
        results = client.search(
            query_embedding=sample_embedding,
            index="test_index",
            top_k=2
        )
        
        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "doc1"
        assert results[0].score == 0.95
        assert results[0].text == "This is answer 1"
        assert results[1].id == "doc2"
        assert results[1].score == 0.87
        
        # Verify ES client was called correctly
        client.es_client.ensure_index_exists.assert_called_once_with("test_index", 384)
        client.es_client.knn_search.assert_called_once_with(
            index_name="test_index",
            query_vector=sample_embedding,
            top_k=2,
            filters=None
        )
    
    def test_search_with_filters(self, client, sample_embedding, sample_es_hits):
        """Test search with filters."""
        client.es_client.ensure_index_exists = Mock()
        client.es_client.knn_search = Mock(return_value=sample_es_hits)
        
        filters = {"metadata.status": "active"}
        results = client.search(
            query_embedding=sample_embedding,
            index="test_index",
            top_k=5,
            filters=filters
        )
        
        assert len(results) == 2
        client.es_client.knn_search.assert_called_once_with(
            index_name="test_index",
            query_vector=sample_embedding,
            top_k=5,
            filters=filters
        )
    
    def test_search_empty_embedding(self, client):
        """Test search with empty embedding raises error."""
        with pytest.raises(ValueError, match="query_embedding cannot be empty"):
            client.search(query_embedding=[], index="test_index")
    
    def test_search_empty_index(self, client, sample_embedding):
        """Test search with empty index name raises error."""
        with pytest.raises(ValueError, match="index name cannot be empty"):
            client.search(query_embedding=sample_embedding, index="")
    
    def test_search_invalid_top_k(self, client, sample_embedding):
        """Test search with invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be positive"):
            client.search(query_embedding=sample_embedding, index="test", top_k=0)
    
    def test_search_invalid_embedding_type(self, client):
        """Test search with invalid embedding type raises error."""
        with pytest.raises(TypeError, match="query_embedding must be a list"):
            client.search(query_embedding="not a list", index="test_index")
        
        with pytest.raises(TypeError, match="query_embedding must contain only numbers"):
            client.search(query_embedding=["not", "numbers"], index="test_index")
    
    def test_search_invalid_index_type(self, client, sample_embedding):
        """Test search with invalid index type raises error."""
        with pytest.raises(TypeError, match="index must be a string"):
            client.search(query_embedding=sample_embedding, index=123)
    
    def test_search_invalid_top_k_type(self, client, sample_embedding):
        """Test search with invalid top_k type raises error."""
        with pytest.raises(TypeError, match="top_k must be an integer"):
            client.search(query_embedding=sample_embedding, index="test", top_k="5")

    def test_search_hybrid_success(self, client, sample_embedding, sample_es_hits):
        """Test successful hybrid search operation."""
        # Mock ES client methods
        client.es_client.ensure_index_exists = Mock()
        client.es_client.hybrid_search = Mock(return_value=sample_es_hits)
        
        # Perform hybrid search
        results = client.search_hybrid(
            query_embedding=sample_embedding,
            query_text="test query",
            index="test_index",
            top_k=2,
        )
        
        # Verify results
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "doc1"
        assert results[1].id == "doc2"
        
        # Verify ES client was called correctly
        client.es_client.ensure_index_exists.assert_called_once_with("test_index", 384)
        client.es_client.hybrid_search.assert_called_once_with(
            index_name="test_index",
            query_vector=sample_embedding,
            query_text="test query",
            top_k=2,
            filters=None,
        )
    
    def test_index_documents_success(self, client):
        """Test successful document indexing."""
        documents = [
            Document(
                id="doc1",
                embedding=[0.1] * 384,
                text="Document 1",
                metadata={"source": "kb"}
            ),
            Document(
                id="doc2",
                embedding=[0.2] * 384,
                text="Document 2",
                metadata={"source": "docs"}
            ),
        ]
        
        # Mock ES client methods
        client.es_client.ensure_index_exists = Mock()
        client.es_client.bulk_index = Mock(return_value=(2, 0))  # Returns (success_count, failed_count)
        
        # Index documents
        count = client.index_documents("test_index", documents)
        
        # Verify
        assert count == 2
        client.es_client.ensure_index_exists.assert_called_once_with("test_index", 384)
        client.es_client.bulk_index.assert_called_once()
        
        # Verify bulk_index was called with correct structure
        call_args = client.es_client.bulk_index.call_args
        assert call_args[0][0] == "test_index"
        es_docs = call_args[0][1]
        assert len(es_docs) == 2
        assert es_docs[0]["_id"] == "doc1"
        assert es_docs[0]["_source"]["text"] == "Document 1"
    
    def test_index_documents_empty_list(self, client):
        """Test indexing empty document list raises error."""
        with pytest.raises(ValueError, match="documents list cannot be empty"):
            client.index_documents("test_index", [])
    
    def test_index_documents_dimension_mismatch(self, client):
        """Test indexing documents with mismatched dimensions raises error."""
        documents = [
            Document(
                id="doc1",
                embedding=[0.1] * 384,  # 384 dimensions
                text="Document 1"
            ),
            Document(
                id="doc2",
                embedding=[0.2] * 256,  # 256 dimensions - mismatch!
                text="Document 2"
            ),
        ]
        
        client.es_client.ensure_index_exists = Mock()
        
        with pytest.raises(ValueError, match="expected 384"):
            client.index_documents("test_index", documents)
    
    def test_health_check_healthy(self, client):
        """Test health check when cluster is healthy."""
        client.es_client.health_check = Mock(return_value=True)
        
        assert client.health_check() is True
    
    def test_health_check_unhealthy(self, client):
        """Test health check when cluster is unhealthy."""
        client.es_client.health_check = Mock(return_value=False)
        
        assert client.health_check() is False
    
    def test_search_exception_handling(self, client, sample_embedding):
        """Test that exceptions during search are properly handled."""
        client.es_client.ensure_index_exists = Mock()
        client.es_client.knn_search = Mock(side_effect=Exception("ES connection error"))
        
        with pytest.raises(Exception, match="ES connection error"):
            client.search(query_embedding=sample_embedding, index="test_index")

    def test_search_bm25_success(self, client, sample_es_hits):
        """Test successful BM25 search."""
        client.es_client.ensure_index_exists = Mock()
        client.es_client.bm25_search = Mock(return_value=sample_es_hits)
        
        results = client.search_bm25(
            query_text="test query",
            index="test_index",
            top_k=2,
        )
        
        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].score == 0.95
        assert results[1].id == "doc2"
        client.es_client.ensure_index_exists.assert_called_once_with("test_index")
        client.es_client.bm25_search.assert_called_once_with(
            index_name="test_index",
            query_text="test query",
            top_k=2,
            filters=None,
        )

    def test_search_bm25_empty_query(self, client):
        """Test BM25 search with empty query raises error."""
        with pytest.raises(ValueError, match="query_text must be a non-empty string"):
            client.search_bm25(query_text="", index="test_index")
        with pytest.raises(ValueError, match="query_text must be a non-empty string"):
            client.search_bm25(query_text="  ", index="test_index")

    def test_search_bm25_invalid_top_k(self, client):
        """Test BM25 search with invalid top_k raises error."""
        with pytest.raises(ValueError, match="top_k must be a positive integer"):
            client.search_bm25(query_text="q", index="test", top_k=0)

    def test_search_hybrid_weighted_uses_strategy(self, client, sample_embedding, sample_es_hits):
        """Test hybrid search with weights uses KNN + BM25 and merges with normalized scores."""
        client.es_client.ensure_index_exists = Mock()
        # Return different order/scores for KNN vs BM25 to test fusion
        knn_hits = [
            {"_id": "doc1", "_score": 0.9, "_source": {"text": "Answer 1", "metadata": {}}},
            {"_id": "doc2", "_score": 0.5, "_source": {"text": "Answer 2", "metadata": {}}},
        ]
        bm25_hits = [
            {"_id": "doc2", "_score": 10.0, "_source": {"text": "Answer 2", "metadata": {}}},
            {"_id": "doc1", "_score": 2.0, "_source": {"text": "Answer 1", "metadata": {}}},
        ]
        client.es_client.knn_search = Mock(return_value=knn_hits)
        client.es_client.bm25_search = Mock(return_value=bm25_hits)
        
        results = client.search_hybrid(
            query_embedding=sample_embedding,
            query_text="test",
            index="test_index",
            top_k=10,
            knn_weight=0.5,
            bm25_weight=0.5,
        )
        
        # Weighted path: should have called KNN and BM25, not hybrid_search
        assert client.es_client.knn_search.called
        assert client.es_client.bm25_search.called
        assert not client.es_client.hybrid_search.called
        # Results should be combined and sorted by final score
        assert len(results) >= 1
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(0 <= r.score <= 1.0 for r in results)

    def test_search_hybrid_invalid_weights(self, client, sample_embedding):
        """Test hybrid search with invalid weights raises error."""
        client.es_client.ensure_index_exists = Mock()
        with pytest.raises(ValueError, match="must sum to 1.0"):
            client.search_hybrid(
                query_embedding=sample_embedding,
                query_text="q",
                index="test_index",
                knn_weight=0.3,
                bm25_weight=0.3,
            )
        with pytest.raises(ValueError, match="knn_weight must be in"):
            client.search_hybrid(
                query_embedding=sample_embedding,
                query_text="q",
                index="test_index",
                knn_weight=1.5,
                bm25_weight=-0.5,
            )


class TestSearchResult:
    """Test cases for SearchResult model."""
    
    def test_from_es_hit(self):
        """Test creating SearchResult from Elasticsearch hit."""
        hit = {
            "_id": "doc123",
            "_score": 0.92,
            "_source": {
                "text": "Answer text",
                "metadata": {"category": "faq"}
            }
        }
        
        result = SearchResult.from_es_hit(hit)
        
        assert result.id == "doc123"
        assert result.score == 0.92
        assert result.text == "Answer text"
        assert result.metadata == {"category": "faq"}
    
    def test_from_es_hit_minimal(self):
        """Test creating SearchResult from minimal hit."""
        hit = {
            "_id": "doc123",
            "_score": 0.5,
            "_source": {
                "text": "Answer"
            }
        }
        
        result = SearchResult.from_es_hit(hit)
        
        assert result.id == "doc123"
        assert result.score == 0.5
        assert result.text == "Answer"
        assert result.metadata is None


class TestDocument:
    """Test cases for Document model."""
    
    def test_to_dict(self):
        """Test converting Document to dictionary."""
        doc = Document(
            id="doc1",
            embedding=[0.1, 0.2, 0.3],
            text="Document text",
            metadata={"source": "kb", "category": "faq"}
        )
        
        doc_dict = doc.to_dict()
        
        assert doc_dict["id"] == "doc1"
        assert doc_dict["embedding"] == [0.1, 0.2, 0.3]
        assert doc_dict["text"] == "Document text"
        assert doc_dict["metadata"] == {"source": "kb", "category": "faq"}
    
    def test_to_dict_no_metadata(self):
        """Test converting Document without metadata."""
        doc = Document(
            id="doc1",
            embedding=[0.1, 0.2],
            text="Document text"
        )
        
        doc_dict = doc.to_dict()
        
        assert "metadata" not in doc_dict
        assert doc_dict["id"] == "doc1"
        assert doc_dict["text"] == "Document text"
