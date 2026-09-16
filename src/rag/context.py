"""
Context Builder and System Prompt Engineering module.
Formats structured telemetry, routing decision facts, and RAG document chunks into an LLM prompt.
"""

from typing import List, Dict, Any, Tuple
from src.rag.ingestion import DocChunk

SYSTEM_PROMPT = """You are the Explainable AI Assistant for an SDN Intelligent Traffic Engineering system.

YOUR RESPONSIBILITIES:
1. Explain network telemetry, ML congestion predictions, TE routing decisions, and What-If simulation results using provided facts.
2. Answer user questions regarding Software-Defined Networking (SDN) concepts and architecture grounded in retrieved project documentation.
3. Clearly cite retrieved source documents.

STRICT RESTRICTIONS:
- You MUST NOT calculate, select, or modify network routing paths or flow tables.
- You MUST NOT modify live network state or Digital Twin parameters.
- You MUST NOT fabricate measurements, packet drop counts, or experimental performance metrics.
- If information is unavailable, explicitly state that it is not available in current telemetry or docs.
"""


def build_prompt_with_context(
    question: str,
    retrieved_chunks: List[Tuple[DocChunk, float]],
    context_data: Dict[str, Any] = None
) -> str:
    """
    Constructs a prompt string containing user question, current simulation facts, and retrieved documentation.
    """
    prompt_parts: List[str] = [f"USER QUESTION: {question}\n"]

    # Add Structured Simulation / Decision Facts if provided
    if context_data:
        prompt_parts.append("=== STRUCTURED LIVE SIMULATION & DECISION FACTS ===")
        for key, val in context_data.items():
            if isinstance(val, list):
                val_str = " -> ".join(str(item) for item in val)
            else:
                val_str = str(val)
            prompt_parts.append(f"- {key}: {val_str}")
        prompt_parts.append("")

    # Add Retrieved RAG Documentation Chunks
    if retrieved_chunks:
        prompt_parts.append("=== RETRIEVED PROJECT DOCUMENTATION CHUNKS ===")
        for chunk, score in retrieved_chunks:
            prompt_parts.append(f"Source: {chunk.source_file} (Section: {chunk.section_header} | Similarity: {score:.2f})")
            prompt_parts.append(f"Content:\n{chunk.content}\n")
    else:
        prompt_parts.append("No relevant project documentation chunks retrieved.\n")

    prompt_parts.append("INSTRUCTION: Provide a clear, technical, and explainable answer based on the facts and docs above.")
    return "\n".join(prompt_parts)
