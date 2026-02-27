"""Shared domain models used across the pipeline."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel


@dataclass
class SearchResult:
    """A single result from the retrieval step."""

    document: str
    code: str
    label: str
    chapter: str
    page_number: int
    similarity: float


class CodeSuggestion(BaseModel):
    """A single CIM-10 code suggestion returned by the generator."""

    code: str
    label: str
    relevance: str
    confidence: Literal["high", "medium", "low"]
    cocoa_info: str = ""


class RAGResponse(BaseModel):
    """Structured output schema for the generation step."""

    codes: list[CodeSuggestion]
    coding_rules: list[str] = []
    warnings: list[str] = []
    related_codes: list[str] = []
