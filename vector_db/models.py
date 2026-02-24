"""
Data models for VectorDB service.

Defines the Document and SearchResult data classes used throughout the service.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """Represents a document to be indexed in Elasticsearch."""
    
    id: str
    embedding: List[float]
    text: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary for Elasticsearch indexing."""
        doc = {
            "id": self.id,
            "embedding": self.embedding,
            "text": self.text,
        }
        if self.metadata:
            doc["metadata"] = self.metadata
        return doc


@dataclass
class SearchResult:
    """Represents a search result returned from semantic search."""
    
    id: str
    score: float
    text: str
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_es_hit(cls, hit: Dict[str, Any]) -> "SearchResult":
        """Create SearchResult from Elasticsearch hit."""
        source = hit.get("_source", {})
        return cls(
            id=hit.get("_id", ""),
            score=hit.get("_score", 0.0),
            text=source.get("text", ""),
            metadata=source.get("metadata"),
        )
