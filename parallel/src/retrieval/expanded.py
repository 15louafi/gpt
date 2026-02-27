"""Query-expansion retriever — wraps another retriever and adds synonym expansion via LLM."""

from openai import OpenAI
from pydantic import BaseModel

from src.models import SearchResult

from .base import Retriever

MODEL = "gpt-4.1"


class _ExpandedQuery(BaseModel):
    terms: list[str]


class ExpandedRetriever:
    """Runs the original query + an LLM-expanded query, then deduplicates."""

    def __init__(self, base: Retriever, openai_client: OpenAI):
        self._base = base
        self._client = openai_client

    def search(self, query: str, n_results: int = 15) -> list[SearchResult]:
        primary = self._base.search(query, n_results=n_results)

        try:
            expanded_query = self._expand(query)
            secondary = self._base.search(expanded_query, n_results=n_results // 2)
        except Exception:
            secondary = []

        return _deduplicate(primary + secondary)

    def _expand(self, query: str) -> str:
        resp = self._client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un expert médical. Génère des termes de recherche pertinents."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Pour le diagnostic '{query}', donne des synonymes médicaux français "
                        "et termes associés."
                    ),
                },
            ],
            response_format=_ExpandedQuery,
            temperature=0.3,
            max_tokens=150,
        )
        parsed = resp.choices[0].message.parsed
        if parsed is None:
            return query
        return f"{query} {' '.join(parsed.terms)}"


def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for r in results:
        key = f"{r.code}_{r.page_number}"
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique.sort(key=lambda r: r.similarity, reverse=True)
    return unique
