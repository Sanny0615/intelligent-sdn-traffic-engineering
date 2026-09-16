"""
AI RAG Explanation route handler.
"""

from fastapi import APIRouter, HTTPException
from src.api.schemas import RAGQueryRequest, RAGQueryResponse
from src.rag.service import RAGService

router = APIRouter(tags=["AI RAG Explanation"])


@router.post("/ai/explain", response_model=RAGQueryResponse, summary="Explain Network State, Routing Decisions, and SDN Concepts")
def explain_network(req: RAGQueryRequest):
    """
    Accepts user question and optional structured simulation facts.
    Retrieves semantic project documentation chunks, queries LLM explanation provider (or local context synthesis),
    and returns human-readable explanation with verified project document citations.
    Does NOT trigger or modify routing decisions.
    """
    rag_service = RAGService.get_instance()
    try:
        result = rag_service.explain(
            question=req.question,
            context_data=req.context_data
        )
        return RAGQueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            provider=result.get("provider", "Local")
        )
    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Explanation generation failed: {str(err)}")
