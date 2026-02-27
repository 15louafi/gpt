"""FastAPI API for the CIM-10 RAG system."""

import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field

from .generation import StructuredGenerator
from .models import RAGResponse
from .pipeline import Pipeline
from .reranking import LLMReranker
from .retrieval import ExpandedRetriever
from .retrieval.semantic import SemanticRetriever

load_dotenv()

_pipeline: Pipeline | None = None


def _build_pipeline(api_key: str) -> Pipeline:
    client = OpenAI(api_key=api_key)
    semantic = SemanticRetriever(client)
    if semantic.count() == 0:
        raise RuntimeError("Vector store is empty. Run 'uv run python -m src.ingest' first.")
    return Pipeline(
        retriever=ExpandedRetriever(semantic, client),
        reranker=LLMReranker(client),
        generator=StructuredGenerator(client),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    _pipeline = _build_pipeline(api_key)
    print("Pipeline ready.")
    yield
    _pipeline = None


app = FastAPI(
    title="CIM-10 RAG - CoCoA",
    description="RAG system for suggesting CIM-10 medical codes from CoCoA.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class QueryRequest(BaseModel):
    text: str = Field(..., max_length=200, examples=["Dyspnée à l'effort"])


class QueryResponse(BaseModel):
    result: RAGResponse
    elapsed_seconds: float


@app.post("/query", response_model=QueryResponse)
async def query_codes(request: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    start = time.time()
    result = _pipeline.query(request.text)
    return QueryResponse(result=result, elapsed_seconds=round(time.time() - start, 2))


@app.get("/health")
async def health():
    return {"status": "healthy", "pipeline_ready": _pipeline is not None}
