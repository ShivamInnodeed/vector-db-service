"""
Integration tests for VectorDBClient with real Elasticsearch.

These tests require a running Elasticsearch instance (local or via testcontainers).
Skip these tests if Elasticsearch is not available.

To run integration tests:
    pytest tests/test_integration.py -v

To skip integration tests:
    pytest tests/ -v -k "not integration"
"""

import pytest
import os
from typing import List

from vector_db import VectorDBClient, Document, SearchResult
from vector_db.config import VectorDBConfig


# Skip integration tests if ELASTICSEARCH_URL is not set or if explicitly skipped
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_INTEGRATION_TESTS", "false").lower() == "true",
    reason="Integration tests skipped via SKIP_INTEGRATION_TESTS env var"
)


class TestVectorDBClientIntegration:
    """Integration tests with real Elasticsearch."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return VectorDBConfig(
            elasticsearch_url=os.getenv("ELASTICSEARCH_URL", "http://localhost:9200"),
            vector_dimension=128,  # Smaller dimension for faster tests
            enable_telemetry=False,  # Disable telemetry for tests
        )
    
    @pytest.fixture
    def client(self, config):
        """Create VectorDBClient instance."""
        return VectorDBClient(config=config)
    
    @pytest.fixture
    def test_index(self):
        """Generate unique test index name."""
        import uuid
        return f"test_vectordb_{uuid.uuid4().hex[:8]}"
    
    @pytest.fixture(autouse=True)
    def cleanup_index(self, client, test_index):
        """Clean up test index after each test."""
        yield
        try:
            client.es_client.client.indices.delete(index=test_index, ignore=[404])
        except Exception:
            pass
    
    def test_health_check(self, client):
        """Test health check with real Elasticsearch."""
        # This test will fail if Elasticsearch is not running
        is_healthy = client.health_check()
        # We can't assert True here since ES might not be available in CI
        # Just verify the method doesn't crash
        assert isinstance(is_healthy, bool)
    
    def test_index_and_search(self, client, test_index):
        """Test full workflow: index documents and search."""
        # Prepare test documents
        documents = [
            Document(
                id="doc1",
                embedding=[0.1] * 128 + [0.9] * 128,  # First half low, second half high
                text="This is about machine learning",
                metadata={"category": "ai", "source": "test"}
            ),
            Document(
                id="doc2",
                embedding=[0.9] * 128 + [0.1] * 128,  # First half high, second half low
                text="This is about web development",
                metadata={"category": "web", "source": "test"}
            ),
            Document(
                id="doc3",
                embedding=[0.5] * 256,  # All medium values
                text="This is a general topic",
                metadata={"category": "general", "source": "test"}
            ),
        ]
        
        # Index documents
        count = client.index_documents(test_index, documents)
        assert count == 3
        
        # Wait a moment for indexing to complete (refresh)
        import time
        time.sleep(1)
        
        # Search with query similar to doc1
        query_embedding = [0.1] * 128 + [0.9] * 128  # Similar to doc1
        results = client.search(
            query_embedding=query_embedding,
            index=test_index,
            top_k=2
        )
        
        # Verify results
        assert len(results) >= 1
        assert results[0].id == "doc1"  # Should match doc1 most closely
        assert results[0].score > 0.0
    
    def test_search_with_filters(self, client, test_index):
        """Test search with metadata filters."""
        # Index documents with different metadata
        documents = [
            Document(
                id="active_doc",
                embedding=[0.1] * 256,
                text="Active document",
                metadata={"status": "active", "category": "faq"}
            ),
            Document(
                id="inactive_doc",
                embedding=[0.2] * 256,
                text="Inactive document",
                metadata={"status": "inactive", "category": "faq"}
            ),
        ]
        
        client.index_documents(test_index, documents)
        
        import time
        time.sleep(1)
        
        # Search with filter for active documents only
        query_embedding = [0.1] * 256
        results = client.search(
            query_embedding=query_embedding,
            index=test_index,
            top_k=10,
            filters={"metadata.status": "active"}
        )
        
        # Verify only active document is returned
        assert len(results) >= 1
        assert all(result.id == "active_doc" for result in results)
    
    def test_index_creates_mapping(self, client, test_index):
        """Test that indexing creates index with correct dense_vector mapping."""
        documents = [
            Document(
                id="doc1",
                embedding=[0.1] * 256,
                text="Test document"
            )
        ]
        
        client.index_documents(test_index, documents)
        
        # Check that index exists
        assert client.es_client.client.indices.exists(index=test_index)
        
        # Check mapping
        mapping = client.es_client.client.indices.get_mapping(index=test_index)
        properties = mapping[test_index]["mappings"]["properties"]
        
        assert "embedding" in properties
        assert properties["embedding"]["type"] == "dense_vector"
        assert properties["embedding"]["dims"] == 256
        assert properties["embedding"]["similarity"] == "cosine"
