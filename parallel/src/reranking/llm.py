"""LLM-based reranker — uses GPT to reorder retrieval results by medical relevance."""

from openai import OpenAI
from pydantic import BaseModel

from src.models import SearchResult

MODEL = "gpt-4.1"


class _RankedIndices(BaseModel):
    indices: list[int]


class LLMReranker:
    def __init__(self, openai_client: OpenAI):
        self._client = openai_client

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = 10
    ) -> list[SearchResult]:
        if len(results) <= top_k:
            return results

        candidates = "\n".join(f"[{i}] {r.code} - {r.label}" for i, r in enumerate(results[:20]))

        try:
            resp = self._client.beta.chat.completions.parse(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Tu es un expert en codage CIM-10."},
                    {
                        "role": "user",
                        "content": (
                            f'Pour le diagnostic "{query}", classe ces codes par pertinence.\n'
                            f"Retourne les indices des {top_k} plus pertinents.\n\n"
                            f"{candidates}"
                        ),
                    },
                ],
                response_format=_RankedIndices,
                temperature=0.0,
                max_tokens=100,
            )
            parsed = resp.choices[0].message.parsed
            if parsed is not None:
                valid = [i for i in parsed.indices if i < len(results)]
                if valid:
                    return [results[i] for i in valid[:top_k]]
        except Exception:
            pass

        return results[:top_k]
