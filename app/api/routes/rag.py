"""
FastAPI Router for Ward Knowledge Base & RAG Operations.
Provides endpoints to query the Ward Handbook knowledge base,
inspect retrieved knowledge chunks, and trigger dynamic re-indexing.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.rag_service import rag_service

router = APIRouter(prefix="/rag", tags=["Ward Knowledge RAG"])


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Citizen's civic question or inquiry in Marathi, English, or Hindi")
    ward_number: Optional[int] = Field(15, description="Target Ward Number (defaults to Ward 15)")
    lang: str = Field("mr", description="Language code: 'mr', 'en', 'hi'")


class RetrievedChunkInfo(BaseModel):
    id: str
    title: str
    ward_number: Optional[int]
    relevance_score: float
    snippet: str


class RAGQueryResponse(BaseModel):
    query: str
    ward_number: Optional[int]
    language: str
    answer: str
    retrieved_chunks: List[RetrievedChunkInfo]
    found_match: bool


@router.post("/query", response_model=RAGQueryResponse)
def query_ward_knowledge(req: RAGQueryRequest):
    """
    Executes a RAG query against the Ward Handbook knowledge base.
    Retrieves matching handbook sections and synthesizes a grounded answer.
    """
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    # Retrieve matching chunks
    raw_matches = rag_service.retrieve(
        query=req.query,
        ward_number=req.ward_number,
        top_k=3,
        min_score=0.10
    )

    chunk_infos = [
        RetrievedChunkInfo(
            id=ch.id,
            title=ch.title,
            ward_number=ch.ward_number,
            relevance_score=round(score, 4),
            snippet=ch.content[:200] + "..." if len(ch.content) > 200 else ch.content
        )
        for ch, score in raw_matches
    ]

    answer = rag_service.query_ward_knowledge(
        query=req.query,
        ward_number=req.ward_number,
        lang=req.lang
    )

    if not answer:
        if req.lang == "mr":
            fallback = "🙏 या विषयाची माहिती सध्या उपलब्ध नाही. अधिक माहितीसाठी कृपया पिंपरी चिंचवड सारथी हेल्पलाइन ०२०-६७३३३३३३ शी संपर्क साधा."
        else:
            fallback = "🙏 Information on this topic is currently not available in the Ward Handbook. Please contact PCMC Sarathi at 020-67333333."
        return RAGQueryResponse(
            query=req.query,
            ward_number=req.ward_number,
            language=req.lang,
            answer=fallback,
            retrieved_chunks=chunk_infos,
            found_match=False
        )

    return RAGQueryResponse(
        query=req.query,
        ward_number=req.ward_number,
        language=req.lang,
        answer=answer,
        retrieved_chunks=chunk_infos,
        found_match=True
    )


@router.post("/reindex")
def reindex_handbooks():
    """
    Rebuilds the vector index from all markdown handbooks located in docs/.
    """
    count = rag_service.reindex()
    return {
        "status": "success",
        "message": f"Successfully reindexed {count} knowledge chunks from Ward Handbooks.",
        "indexed_chunks": count
    }


@router.get("/status")
def get_rag_status():
    """
    Returns the current index statistics (number of indexed chunks and files).
    """
    return {
        "status": "ready",
        "total_chunks": len(rag_service.store.chunks),
        "vocabulary_size": len(rag_service.store.vocab),
        "source_files": list({ch.source_file for ch in rag_service.store.chunks})
    }
