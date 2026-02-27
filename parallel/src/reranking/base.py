"""Reranker protocol — any reranking strategy must implement this."""

from typing import Protocol

from src.models import SearchResult


class Reranker(Protocol):
    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 10
    ) -> list[SearchResult]: ...
