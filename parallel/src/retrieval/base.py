"""Retriever protocol — any retrieval strategy must implement this."""

from typing import Protocol

from src.models import SearchResult


class Retriever(Protocol):
    def search(self, query: str, n_results: int = 15) -> list[SearchResult]: ...
