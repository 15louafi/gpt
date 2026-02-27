"""RAG Pipeline — thin orchestrator that wires retrieval, reranking, and generation."""

from src.generation.base import Generator
from src.models import RAGResponse
from src.reranking.base import Reranker
from src.retrieval.base import Retriever


class Pipeline:
    def __init__(self, retriever: Retriever, reranker: Reranker, generator: Generator):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def query(self, text: str, n_retrieve: int = 15, top_k: int = 10) -> RAGResponse:
        results = self.retriever.search(text, n_results=n_retrieve)
        reranked = self.reranker.rerank(text, results, top_k=top_k)
        return self.generator.generate(text, reranked)
