"""Generator protocol — any generation strategy must implement this."""

from typing import Protocol

from src.models import RAGResponse, SearchResult


class Generator(Protocol):
    def generate(self, query: str, context: list[SearchResult]) -> RAGResponse: ...
