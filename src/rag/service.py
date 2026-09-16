"""
RAG Service orchestration module.
Ingests project documentation, retrieves semantic chunks, constructs contexts, and calls LLM explanation providers.
"""

from typing import Dict, Any, List, Tuple, Optional
import os

from src.rag.ingestion import load_and_chunk_docs, DocChunk
from src.rag.vectorstore import LocalVectorStore
from src.rag.llm import get_llm_provider
from src.rag.context import SYSTEM_PROMPT, build_prompt_with_context


class RAGService:
    """
    RAG Service orchestrating document ingestion, semantic retrieval, and AI explanation generation.
    """

    _instance: Optional["RAGService"] = None

    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.vector_store = LocalVectorStore()
        self.ingest_docs(self.docs_dir)

    def ingest_docs(self, docs_dir: str = "docs") -> int:
        """Ingests all markdown documents in docs_dir and builds vector index."""
        if not os.path.exists(docs_dir):
            return 0
        chunks = load_and_chunk_docs(docs_dir=docs_dir)
        count = self.vector_store.build_index(chunks)
        return count

    def retrieve(self, question: str, top_k: int = 3) -> List[Tuple[DocChunk, float]]:
        """Retrieves top_k semantic documentation chunks for a question."""
        return self.vector_store.search(question, top_k=top_k)

    def explain(
        self,
        question: str,
        context_data: Optional[Dict[str, Any]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Orchestrates full RAG explanation workflow:
        1. Retrieves relevant project docs chunks.
        2. Formulates prompt with structured simulation facts.
        3. Calls configured LLM provider (or local fallback).
        4. Returns answer with verified document citations.
        """
        retrieved = self.retrieve(question, top_k=top_k)

        # Extract unique source files
        sources = list(dict.fromkeys([chunk.source_file for chunk, _ in retrieved]))
        if not sources:
            sources = ["docs/architecture.md"]

        prompt = build_prompt_with_context(question, retrieved, context_data)
        provider = get_llm_provider()
        res = provider.generate_explanation(prompt, SYSTEM_PROMPT)

        chunk_info: List[Dict[str, Any]] = [
            {
                "chunk_id": chunk.chunk_id,
                "source_file": chunk.source_file,
                "section_header": chunk.section_header,
                "similarity_score": score,
                "content_snippet": chunk.content[:150] + "..."
            }
            for chunk, score in retrieved
        ]

        return {
            "question": question,
            "answer": res["answer"],
            "sources": sources,
            "provider": res.get("provider", "Local"),
            "retrieved_chunks": chunk_info
        }

    @classmethod
    def get_instance(cls, docs_dir: str = "docs") -> "RAGService":
        """Singleton getter for RAGService."""
        if cls._instance is None:
            cls._instance = RAGService(docs_dir=docs_dir)
        return cls._instance
