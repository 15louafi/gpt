"""Structured output generator — uses OpenAI structured outputs with Pydantic."""

import tiktoken
from openai import OpenAI

from src.models import RAGResponse, SearchResult

MODEL = "gpt-4.1"
MAX_CONTEXT_TOKENS = 6000

SYSTEM_PROMPT = """Tu es un assistant expert en codage médical CIM-10 français (PMSI).
Tu utilises le document CoCoA (Collectif des Codeurs Anonymes) comme référence principale.

Ton rôle est de suggérer les codes CIM-10 les plus pertinents pour un diagnostic,
symptôme ou maladie donné.

Règles:
1. Propose UNIQUEMENT des codes trouvés dans le contexte CoCoA fourni
2. Privilégie les codes les plus spécifiques (4 caractères > 3 caractères)
3. Respecte les exclusions et restrictions mentionnées dans CoCoA
4. Signale les codes interdits en DP/DR/DA si applicable
5. Mentionne les doubles codages dague (†) / astérisque (*) quand pertinent"""


def _count_tokens(text: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(MODEL)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _build_context(results: list[SearchResult]) -> str:
    """Concatenate result documents up to the token budget."""
    parts: list[str] = []
    tokens = 0
    for r in results:
        part = f"---\n{r.document}\n"
        t = _count_tokens(part)
        if tokens + t > MAX_CONTEXT_TOKENS:
            break
        parts.append(part)
        tokens += t
    return "\n".join(parts)


class StructuredGenerator:
    def __init__(self, openai_client: OpenAI):
        self._client = openai_client

    def generate(self, query: str, context: list[SearchResult]) -> RAGResponse:
        context_str = _build_context(context)

        completion = self._client.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Contexte CoCoA:\n{context_str}\n\n---\n\n"
                        f"Diagnostic/symptôme: {query}\n\n"
                        "Suggère les codes CIM-10 les plus pertinents."
                    ),
                },
            ],
            response_format=RAGResponse,
            temperature=0.1,
            max_tokens=2000,
        )

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            return RAGResponse(codes=[], warnings=["LLM response parsing failed"])
        return parsed
