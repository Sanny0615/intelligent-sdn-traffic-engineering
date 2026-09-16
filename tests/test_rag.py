"""
Unit test suite for Phase 8 RAG + LLM Explanation Assistant.
"""

import os
import pytest
from fastapi.testclient import TestClient

from src.rag.ingestion import load_and_chunk_docs, DocChunk
from src.rag.vectorstore import LocalVectorStore
from src.rag.llm import MockFallbackLLMProvider, GroqLLMProvider, get_llm_provider
from src.rag.context import build_prompt_with_context, SYSTEM_PROMPT
from src.rag.service import RAGService
from src.api.main import app

client = TestClient(app)


def test_document_ingestion_and_chunking():
    """Verify loading and splitting of project markdown docs."""
    chunks = load_and_chunk_docs(docs_dir="docs", chunk_size=300)
    assert len(chunks) > 0

    first = chunks[0]
    assert isinstance(first, DocChunk)
    assert first.source_file.startswith("docs/")
    assert len(first.content) > 0


def test_vector_store_build_and_search():
    """Verify LocalVectorStore indexing and similarity search."""
    chunks = [
        DocChunk("c1", "NetworkX Engine handles packet simulation.", "docs/architecture.md", "Simulation"),
        DocChunk("c2", "Random Forest predicts link congestion.", "docs/ml-strategy.md", "ML Strategy"),
        DocChunk("c3", "What-If Digital Twin deep-clones network state.", "docs/architecture.md", "Digital Twin")
    ]
    store = LocalVectorStore()
    count = store.build_index(chunks)
    assert count == 3

    results = store.search("Tell me about ML congestion prediction", top_k=2)
    assert len(results) == 2
    top_chunk, score = results[0]
    assert isinstance(top_chunk, DocChunk)
    assert isinstance(score, float)
    assert score >= -1.0



def test_context_builder_system_prompt():
    """Verify prompt formatting with structured facts and retrieved RAG chunks."""
    chunks = [(DocChunk("c1", "Sample text", "docs/architecture.md", "Header"), 0.95)]
    facts = {"selected_path": ["H1", "S1", "S2", "S3", "H3"], "demand_mbps": 400.0}

    prompt = build_prompt_with_context("Why was this flow rerouted?", chunks, facts)
    assert "USER QUESTION: Why was this flow rerouted?" in prompt
    assert "STRUCTURED LIVE SIMULATION & DECISION FACTS" in prompt
    assert "H1 -> S1 -> S2 -> S3 -> H3" in prompt
    assert "docs/architecture.md" in prompt


def test_llm_provider_mock_fallback():
    """Verify MockFallbackLLMProvider produces structured response without external network calls."""
    provider = MockFallbackLLMProvider()
    res = provider.generate_explanation("Explain routing", SYSTEM_PROMPT)

    assert "answer" in res
    assert "AI Explanation Assistant" in res["answer"]
    assert res["provider"] == "Local Context Synthesis"


def test_rag_service_explain():
    """Verify RAGService full pipeline execution."""
    rag_service = RAGService(docs_dir="docs")
    res = rag_service.explain("What is ML-Assisted Predictive Traffic Engineering?")

    assert "question" in res
    assert "answer" in res
    assert "sources" in res
    assert isinstance(res["sources"], list)
    assert len(res["sources"]) > 0


def test_api_explain_endpoint():
    """Verify POST /api/v1/ai/explain API endpoint."""
    payload = {
        "question": "What is ML-Assisted Predictive Traffic Engineering?",
        "context_data": {"selected_path": ["H1", "S1", "S2", "S3", "H3"]}
    }
    response = client.post("/api/v1/ai/explain", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == payload["question"]
    assert "answer" in data
    assert len(data["sources"]) > 0
