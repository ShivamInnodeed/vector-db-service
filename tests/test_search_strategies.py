"""
Unit tests for search strategies and hybrid score normalization.
"""

import pytest
from unittest.mock import Mock

from vector_db.config import HybridSearchConfig, _validate_weights
from vector_db.models import SearchResult
from vector_db.search.hybrid import _merge_and_combine, _min_max_normalize
from vector_db.search import KNNSearchStrategy, BM25SearchStrategy, HybridSearchStrategy


class TestHybridSearchConfig:
    """Tests for HybridSearchConfig and weight validation."""

    def test_valid_weights(self):
        HybridSearchConfig(knn_weight=0.5, bm25_weight=0.5)
        HybridSearchConfig(knn_weight=1.0, bm25_weight=0.0)
        HybridSearchConfig(knn_weight=0.0, bm25_weight=1.0)
        HybridSearchConfig(knn_weight=0.7, bm25_weight=0.3)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            HybridSearchConfig(knn_weight=0.5, bm25_weight=0.5 + 0.1)
        with pytest.raises(ValueError, match="must sum to 1.0"):
            _validate_weights(0.3, 0.3)

    def test_weights_in_range(self):
        with pytest.raises(ValueError, match="knn_weight must be in"):
            _validate_weights(1.5, -0.5)
        with pytest.raises(ValueError, match="bm25_weight must be in"):
            _validate_weights(0.5, 1.2)


class TestMinMaxNormalize:
    """Tests for score normalization."""

    def test_normalize_basic(self):
        out = _min_max_normalize([1.0, 2.0, 3.0])
        assert out == [0.0, 0.5, 1.0]

    def test_normalize_single_value(self):
        out = _min_max_normalize([5.0])
        assert out == [1.0]

    def test_normalize_all_equal(self):
        out = _min_max_normalize([2.0, 2.0, 2.0])
        assert out == [1.0, 1.0, 1.0]

    def test_normalize_empty(self):
        assert _min_max_normalize([]) == []


class TestMergeAndCombine:
    """Tests for hybrid merge and weighted combination."""

    def test_merge_combine_formula(self):
        knn_results = [
            SearchResult(id="a", score=0.2, text="A", metadata=None),
            SearchResult(id="b", score=1.0, text="B", metadata=None),
        ]
        bm25_results = [
            SearchResult(id="a", score=10.0, text="A", metadata=None),
            SearchResult(id="b", score=2.0, text="B", metadata=None),
        ]
        combined = _merge_and_combine(knn_results, bm25_results, 0.5, 0.5)
        # KNN norms: (0.2->0, 1.0->1); BM25 norms: (2->0, 10->1) -> (a: 0, b: 1) for BM25, (a: 0, b: 1) for KNN
        # Actually min-max: knn [0.2, 1.0] -> (0.2-0.2)/(1-0.2)=0, (1-0.2)/(1-0.2)=1 -> [0, 1]
        # bm25 [10, 2] -> min=2, max=10 -> (10-2)/8=1, (2-2)/8=0 -> for a: 10->1, for b: 2->0
        # So doc a: 0.5*0 + 0.5*1 = 0.5; doc b: 0.5*1 + 0.5*0 = 0.5. Order undefined for tie.
        assert len(combined) == 2
        ids = {r.id for r in combined}
        assert ids == {"a", "b"}
        scores = {r.id: r.score for r in combined}
        assert 0.49 <= scores["a"] <= 0.51
        assert 0.49 <= scores["b"] <= 0.51

    def test_merge_only_knn(self):
        knn_results = [SearchResult(id="a", score=0.8, text="A", metadata=None)]
        bm25_results = []
        combined = _merge_and_combine(knn_results, bm25_results, 0.5, 0.5)
        assert len(combined) == 1
        assert combined[0].id == "a"
        assert combined[0].score == 0.5  # 0.5*1.0 + 0.5*0

    def test_merge_only_bm25(self):
        knn_results = []
        bm25_results = [SearchResult(id="b", score=3.0, text="B", metadata=None)]
        combined = _merge_and_combine(knn_results, bm25_results, 0.5, 0.5)
        assert len(combined) == 1
        assert combined[0].id == "b"
        assert combined[0].score == 0.5


class TestKNNSearchStrategy:
    """Tests for KNN strategy."""

    def test_execute_requires_embedding(self):
        strategy = KNNSearchStrategy()
        mock_es = Mock()
        with pytest.raises(ValueError, match="query_embedding"):
            strategy.execute(mock_es, index="i", top_k=5, query_text="q")

    def test_execute_calls_knn_search(self, sample_es_hits):
        strategy = KNNSearchStrategy()
        mock_es = Mock()
        mock_es.knn_search.return_value = sample_es_hits
        results = strategy.execute(
            mock_es,
            index="idx",
            top_k=2,
            query_embedding=[0.1] * 4,
        )
        assert len(results) == 2
        mock_es.knn_search.assert_called_once_with(
            index_name="idx",
            query_vector=[0.1] * 4,
            top_k=2,
            filters=None,
        )


class TestBM25SearchStrategy:
    """Tests for BM25 strategy."""

    def test_execute_requires_query_text(self):
        strategy = BM25SearchStrategy()
        mock_es = Mock()
        with pytest.raises(ValueError, match="query_text"):
            strategy.execute(mock_es, index="i", top_k=5, query_embedding=[0.1])

    def test_execute_calls_bm25_search(self, sample_es_hits):
        strategy = BM25SearchStrategy()
        mock_es = Mock()
        mock_es.bm25_search.return_value = sample_es_hits
        results = strategy.execute(
            mock_es,
            index="idx",
            top_k=2,
            query_text="hello world",
        )
        assert len(results) == 2
        mock_es.bm25_search.assert_called_once_with(
            index_name="idx",
            query_text="hello world",
            top_k=2,
            filters=None,
        )


@pytest.fixture
def sample_es_hits():
    return [
        {"_id": "doc1", "_score": 0.95, "_source": {"text": "Answer 1", "metadata": {}}},
        {"_id": "doc2", "_score": 0.87, "_source": {"text": "Answer 2", "metadata": {}}},
    ]
