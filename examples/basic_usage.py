"""
Basic usage example for VectorDB service.

This demonstrates how to use the VectorDB service in a Langgraph Node 2 context,
for Knowledge Ingestion flow, and for RAG (retrieval + LLM) pipelines.

Run from project root:
  cd vector_db_service
  pip install -e .   # install the package once
  python examples/basic_usage.py

Or run with PYTHONPATH so the vector_db package is found:
  cd vector_db_service && PYTHONPATH=. python examples/basic_usage.py
"""

import sys
from pathlib import Path
from typing import List

# When running as a script, add project root so "vector_db" package is found
if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from vector_db import VectorDBClient, Document, SearchResult


# =============================================================================
# Quick reference: RAG flow with our VectorDB module (copy-paste friendly)
# =============================================================================
#
# # Step 1: Query processing (your embedding service)
# query_text = "how to reset password"
# query_embedding = embedding_service.encode(query_text)  # → vector
#
# # Step 2: Our VectorDB module (KNN + BM25 → hybrid → top k)
# from vector_db import VectorDBClient
# client = VectorDBClient()
#
# # Hybrid: query_text → ES (BM25) + query_embedding → ES (KNN) → combine → top k
# results = client.search_hybrid(
#     query_embedding=query_embedding,
#     query_text=query_text,
#     index="rag_chunks",
#     top_k=3,
# )
#
# # KNN/BM25 weights (must sum to 1.0): pass as keyword arguments
# results = client.search_hybrid(
#     query_embedding=query_embedding,
#     query_text=query_text,
#     index="rag_chunks",
#     top_k=3,
#     knn_weight=0.6,   # weight for semantic (vector) score
#     bm25_weight=0.4,  # weight for keyword score
# )
#
# # Step 3: Top k results + query → LLM (your RAG code)
# context = "\n".join([r.text for r in results])
# prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
# output = llm.generate(prompt)  # → final answer
#
# =============================================================================


def example_rag_flow():
    """
    End-to-end RAG flow: query → embedding → VectorDB (hybrid) → top k → LLM.
    Use this pattern to integrate our module into your RAG pipeline.
    """
    print("=== Example: RAG Flow (Query → VectorDB → LLM) ===")

    client = VectorDBClient()

    # Seed a few sample chunks so the example returns results (skip in production)
    sample_docs = [
        Document(
            id="chunk_1",
            embedding=[0.12] * 384,  # placeholder; in production use real embeddings
            text="To reset your password, go to Settings > Security > Reset password and follow the steps.",
            metadata={"source": "faq"},
        ),
        Document(
            id="chunk_2",
            embedding=[0.11] * 384,
            text="Password reset requires your email. We send a reset link to your registered email address.",
            metadata={"source": "docs"},
        ),
        Document(
            id="chunk_3",
            embedding=[0.13] * 384,
            text="How to reset password: open the app, tap Profile, then Security, then Reset password.",
            metadata={"source": "kb"},
        ),
    ]
    client.index_documents("rag_chunks", sample_docs, refresh=True)
    print("Indexed 3 sample chunks into rag_chunks.")

    # Step 1: Query processing (your embedding service)
    query_text = "how to reset password"
    # In production: query_embedding = embedding_service.encode(query_text)
    query_embedding = [0.1] * 384  # placeholder; replace with real embedding

    # Step 2: Run all 3 search types and log results
    top_k = 3

    # --- 1. KNN only (semantic / vector search)
    results_knn = client.search(
        query_embedding=query_embedding,
        index="rag_chunks",
        top_k=top_k,
    )
    print("\n--- 1. KNN (semantic) search results ---")
    print(f"Count: {len(results_knn)}")
    for i, r in enumerate(results_knn, 1):
        print(f"  {i}. id={r.id} score={r.score:.4f} text={r.text[:60]}...")
    print()

    # --- 2. BM25 only (keyword search)
    results_bm25 = client.search_bm25(
        query_text=query_text,
        index="rag_chunks",
        top_k=top_k,
    )
    print("--- 2. BM25 (keyword) search results ---")
    print(f"Count: {len(results_bm25)}")
    for i, r in enumerate(results_bm25, 1):
        print(f"  {i}. id={r.id} score={r.score:.4f} text={r.text[:60]}...")
    print()

    # --- 3. Hybrid (KNN + BM25, weighted)
    results_hybrid = client.search_hybrid(
        query_embedding=query_embedding,
        query_text=query_text,
        index="rag_chunks",
        top_k=top_k,
        knn_weight=0.6,
        bm25_weight=0.4,
    )
    print("--- 3. Hybrid (KNN + BM25 weighted) search results ---")
    print(f"Count: {len(results_hybrid)}")
    for i, r in enumerate(results_hybrid, 1):
        print(f"  {i}. id={r.id} score={r.score:.4f} text={r.text[:60]}...")
    print()

    # Step 3: Use hybrid results for RAG (you can switch to results_knn or results_bm25 if needed)
    results = results_hybrid
    context = "\n".join([r.text for r in results])
    prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
    print("--- RAG prompt (first 200 chars) ---")
    print(f"{prompt[:200]}...")
    print(f"\nRetrieved {len(results)} chunks for RAG (from hybrid).")


def example_search():
    """Example: Semantic search (Langgraph Node 2)."""
    print("=== Example: Semantic Search ===")
    
    # Initialize client
    client = VectorDBClient()
    
    # Simulate a query embedding (in real usage, this comes from Embedding Model Inferencing service)
    query_embedding = [0.1] * 384  # Replace with actual embedding
    
    # Perform search
    results: list[SearchResult] = client.search(
        query_embedding=query_embedding,
        index="livechat_answers",
        top_k=5
    )
    
    # Process results
    print(f"Found {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. ID: {result.id}")
        print(f"   Score: {result.score:.4f}")
        print(f"   Text: {result.text[:100]}...")  # Truncate for display
        if result.metadata:
            print(f"   Metadata: {result.metadata}")


def example_search_with_filters():
    """Example: Search with filters."""
    print("\n=== Example: Search with Filters ===")
    
    client = VectorDBClient()
    query_embedding = [0.1] * 384
    
    # Search with filters
    results = client.search(
        query_embedding=query_embedding,
        index="livechat_answers",
        top_k=10,
        filters={
            "metadata.status": "active",
            "metadata.category": "faq"
        }
    )
    
    print(f"Found {len(results)} active FAQ results")


def example_index_documents():
    """Example: Index documents (Knowledge Ingestion Flow)."""
    print("\n=== Example: Index Documents ===")
    
    client = VectorDBClient()
    
    # Prepare documents (embeddings should be pre-computed)
    documents = [
        Document(
            id="doc_001",
            embedding=[0.1] * 384,  # Replace with actual embedding from Embedding Model Inferencing service
            text="This is a frequently asked question about product features.",
            metadata={
                "source": "kb",
                "category": "faq",
                "status": "active",
                "created_at": "2024-01-01"
            }
        ),
        Document(
            id="doc_002",
            embedding=[0.2] * 384,
            text="Troubleshooting guide for common issues.",
            metadata={
                "source": "docs",
                "category": "troubleshooting",
                "status": "active"
            }
        ),
    ]
    
    # Index documents
    count = client.index_documents(
        index="livechat_answers",
        documents=documents
    )
    
    print(f"Successfully indexed {count} documents")


def example_health_check():
    """Example: Health check."""
    print("\n=== Example: Health Check ===")
    
    client = VectorDBClient()
    is_healthy = client.health_check()
    
    if is_healthy:
        print("Elasticsearch cluster is healthy")
    else:
        print("Elasticsearch cluster is not healthy")


def example_langgraph_integration():
    """Example: Integration with Langgraph Node 2."""
    print("\n=== Example: Langgraph Integration ===")
    
    # This is how you would use it in a Langgraph node
    def node_2_semantic_search(state):
        """Node 2: Semantically search on ES for similar query and retrieve answers"""
        client = VectorDBClient()
        
        # Get query embedding from state (computed in Node 3: Query Embedding)
        query_embedding = state.get("query_embedding")
        
        if not query_embedding:
            state["search_results"] = []
            state["similar_query_found"] = False
            return state
        
        # Search for similar queries
        results = client.search(
            query_embedding=query_embedding,
            index="livechat_answers",
            top_k=5
        )
        
        # Update state with results
        state["search_results"] = results
        state["similar_query_found"] = len(results) > 0
        
        # Decision: If similar query found, use Node 3; else use Node 4
        if state["similar_query_found"]:
            print(f"Found {len(results)} similar queries, proceeding to Node 3")
        else:
            print("No similar queries found, proceeding to Node 4 (generic response)")
        
        return state
    
    # Simulate state
    state = {
        "query_embedding": [0.1] * 384,
        "query_text": "How do I reset my password?"
    }
    
    # Execute node
    state = node_2_semantic_search(state)
    
    print(f"State updated: similar_query_found={state['similar_query_found']}")
    print(f"Number of results: {len(state.get('search_results', []))}")


if __name__ == "__main__":
    # These examples need a running Elasticsearch. Set ELASTICSEARCH_URL if not on localhost:9200.
    print("VectorDB Service - Usage Examples")
    print("=" * 50)

    try:
        example_health_check()
    except Exception as e:
        print(f"\nError: {e}")

    try:
        client = VectorDBClient()
        if client.health_check():
            print("\nElasticsearch is ready. Running RAG flow example...")
            example_rag_flow()
        else:
            print(
                "\nElasticsearch is not healthy. Start it (e.g. docker run -p 9200:9200 -e discovery.type=single-node elasticsearch:9.2.1), "
                "then run this script again."
            )
    except Exception as e:
        print(
            "\nElasticsearch is not available. Start it (e.g. docker run -p 9200:9200 -e discovery.type=single-node elasticsearch:9.2.1) "
            "or set ELASTICSEARCH_URL, then run this script again."
        )
        print(f"Details: {e}")
