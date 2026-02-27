# CIM-10 RAG System - CoCoA

RAG system for suggesting ICD-10 medical codes (French CIM-10) based on the **CoCoA** reference document using GPT-4.1.

**Development time: ~3-4 hours** (functional prototype with pragmatic design)

## Architecture

```
User Input → Query Expansion (GPT-4.1) → Semantic Search (ChromaDB)
    → Re-ranking (GPT-4.1) → Generation (GPT-4.1) → Structured JSON Output
```

### Components

| Module | Role |
|--------|------|
| `src/pdf_parser.py` | Extract CIM-10 codes from PDF (regex + PyMuPDF) |
| `src/models.py` | Shared Pydantic models (`SearchResult`, `RAGResponse`) |
| `src/pipeline.py` | Thin orchestrator wiring retrieval → reranking → generation |
| `src/retrieval/` | Retriever protocol + semantic (ChromaDB) and expanded (query expansion) implementations |
| `src/reranking/` | Reranker protocol + LLM-based reranking implementation |
| `src/generation/` | Generator protocol + structured output generation implementation |
| `src/api.py` | FastAPI REST API |
| `src/app.py` | Streamlit UI |
| `src/evaluate.py` | Metrics: Hit@k, Recall, Latency |
| `src/ingest.py` | PDF parsing + vector index building |

## Installation

### Prerequisites
- Python 3.13+, [uv](https://docs.astral.sh/uv/), OpenAI API key

```bash
# 1. Install dependencies
uv sync

# 2. Configure API key
echo 'OPENAI_API_KEY=sk-...' > .env

# 3. Parse PDF + build vector index (~2-5 min, ~13.7k chunks)
uv run python -m src.ingest

# 4. Launch Streamlit UI
uv run streamlit run src/app.py

# Or REST API
uv run uvicorn src.api:app --reload --port 8000

# Or evaluation
uv run python -m src.evaluate
```

## API Usage

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"text": "Dyspnee a l effort"}'
```

Response (structured output via Pydantic):
```json
{
  "codes": [
    {
      "code": "R06.0",
      "label": "Dyspnée",
      "relevance": "Primary code for dyspnea (respiratory difficulty)",
      "confidence": "high",
      "cocoa_info": "Includes: Orthopnea, Shortness of breath. Excludes: transient tachypnea of newborn"
    }
  ],
  "coding_rules": ["..."],
  "warnings": ["..."],
  "related_codes": ["..."]
}
```

## Technical Choices (Time-boxed Approach)

### 1. Simplified Chunking

**Choice**: One chunk = one CIM-10 code + its raw context from the PDF.

**Justification**: The CoCoA PDF is structured with one code per section (P/R/A + code + label + context). The parser detects codes via regex (`[A-Z]\d{2}(\.\d{1,2})?`) and captures all text until the next code. No complex parsing of exclusions/inclusions/notes: everything is kept in `full_text` which is embedded as-is.

**Result**: 13,693 chunks extracted in ~130 lines of code.

**Alternative considered**: Fine-grained parsing of "Comprend", "À l'exclusion" sections — Rejected as too time-consuming, and the LLM can interpret raw text.

### 2. Modular RAG Pipeline

**Choice**: Protocol-based architecture with separate retrieval, reranking, and generation modules.

**Justification**:
- **Retriever protocol** (`src/retrieval/base.py`): Defines a common interface; `SemanticRetriever` (ChromaDB) and `ExpandedRetriever` (query expansion decorator) implement it
- **Reranker protocol** (`src/reranking/base.py`): `LLMReranker` uses GPT-4.1 to reorder results by medical relevance
- **Generator protocol** (`src/generation/base.py`): `StructuredGenerator` uses OpenAI structured outputs with Pydantic
- **Pipeline** (`src/pipeline.py`): Thin ~20-line orchestrator that wires the three stages together

**Pipeline stages**:
1. **Query expansion**: "Dyspnée" → "Dyspnée, essoufflement, difficulté respiratoire, orthopnée" via GPT-4.1
2. **Retrieval**: Vector search (cosine similarity) with OpenAI `text-embedding-3-small` embeddings
3. **Re-ranking**: GPT-4.1 re-orders top results by medical relevance
4. **Generation**: GPT-4.1 generates structured JSON with codes + explanations

**Cost**: 3 API calls/query → ~10-20s latency.

### 3. Structured Outputs

**Choice**: Use OpenAI's structured outputs with Pydantic models instead of `response_format: json_object`.

**Justification**:
- **Type safety**: Pydantic models define the exact schema (`RAGResponse` with nested `CodeSuggestion`)
- **No parsing errors**: OpenAI guarantees valid output matching the schema
- **Better reliability**: No need for try/catch around JSON parsing

**Implementation**: `client.beta.chat.completions.parse()` with `response_format=RAGResponse` Pydantic model.

### 4. Deliberate Simplifications

To meet the 3-4 hour constraint, several simplifications were made:

| Aspect | Simplified | Alternative (not implemented) |
|--------|-----------|-------------------------------|
| **PDF Parsing** | Simple regex, raw full_text | Parse XML structure, extract tables |
| **Metadata** | Code + label + chapter only | Fine extraction of exclusions/inclusions/notes |
| **Vector DB** | Single ChromaDB collection | Separate collections for codes/rules, metadata filters |
| **Re-ranking** | Simple LLM (list of indices) | Fine-tuned cross-encoder |
| **Evaluation** | 15 cases, approximate codes | Gold standard annotated by medical coders |
| **Caching** | None | Redis cache query → results |

### 5. Tech Stack

| Tool | Reason |
|------|--------|
| **PyMuPDF** | Fast PDF text extraction |
| **ChromaDB** | Local vector store, zero config |
| **OpenAI text-embedding-3-small** | Best quality/price for French medical text |
| **GPT-4.1** | Structured outputs + medical reasoning |
| **FastAPI + Streamlit** | Rapid API + UI prototyping |
| **uv** | Lock files, reproducible builds |

## Evaluation

**Validation set**: 15 cases (symptoms, metabolic, cardio, respiratory, infectious)

**Metrics**:
- **Hit@3**: Is the expected code in the top 3 suggestions?
- **Hit@5**: In the top 5?
- **Recall**: Fraction of expected codes found
- **Latency**: End-to-end response time

Run evaluation: `uv run python -m src.evaluate`

### Benchmark Results

```
Hit@3:        100.00%
Hit@5:        100.00%
Mean Recall:  100.00%
Mean Latency: 12.78s
Total Cases:  15
```

**Note**: Perfect scores (100%) are due to the small validation set (15 cases) with approximate expected codes. A rigorous evaluation would require a larger gold standard annotated by professional medical coders (DIM).

## Limitations

1. **Imperfect PDF extraction**: Complex tables, decision trees (images), DP/DR/DA formatting lost
2. **High latency**: 3 GPT-4.1 calls/query → 10-20s (expansion + re-ranking could be optional in prod)
3. **No medical validation**: Does not replace a medical coder (DIM)
4. **Ambiguous inputs**: Short diagnoses without context → broad suggestions
5. **Mono-diagnosis**: No handling of comorbidities/hospitalization context
6. **Limited evaluation**: 15 cases, "expected" codes are approximate (no gold standard)

## Improvement Suggestions

**Short term** (1-2 days):
- Hybrid search: BM25 + vectors for exact code matching
- LRU cache (Redis) for frequent queries
- Streaming for progressive feedback

**Medium term** (1-2 weeks):
- Fine-tune embeddings on CIM-10 vocabulary
- Cross-encoder re-ranking (replaces LLM reranker, faster)
- Fine-grained CoCoA parsing: exclusions, inclusions, code hierarchy
- Gold standard: 100+ cases annotated by professional coders

**Long term** (1+ month):
- Vision LLM for decision trees (PDF images)
- Graph RAG: model code relationships (exclusions, double coding)
- Multi-diagnosis: coherence + incompatibility detection
- RLHF with medical coder feedback

## Project Structure

```
cocoa-rag/
├── CoCoA.pdf                       # CoCoA 2023 source document
├── .env                            # OPENAI_API_KEY
├── pyproject.toml                  # uv config + dependencies
├── uv.lock
├── README.md
├── data/
│   ├── chunks.json                 # 13,693 extracted chunks
│   ├── chroma_db/                  # Vector index
│   └── evaluation_results.json
└── src/
    ├── models.py                   # Shared Pydantic models
    ├── pipeline.py                 # Thin orchestrator (~20 lines)
    ├── pdf_parser.py               # PDF extraction (~130 lines)
    ├── ingest.py                   # Ingestion script
    ├── api.py                      # FastAPI REST API
    ├── app.py                      # Streamlit UI
    ├── evaluate.py                 # Evaluation framework
    ├── retrieval/
    │   ├── base.py                 # Retriever protocol
    │   ├── semantic.py             # ChromaDB + OpenAI embeddings
    │   └── expanded.py             # Query expansion decorator
    ├── reranking/
    │   ├── base.py                 # Reranker protocol
    │   └── llm.py                  # LLM-based reranking
    └── generation/
        ├── base.py                 # Generator protocol
        └── structured.py           # Structured output generation
```

## Key Features

- **Modular architecture**: Protocol-based interfaces for retrieval, reranking, and generation
- **Structured outputs**: Type-safe JSON generation with Pydantic models
- **Multi-stage RAG**: Query expansion + semantic search + LLM re-ranking
- **13,693 CIM-10 codes indexed**: Complete CoCoA 2023 coverage
- **Production-ready API**: FastAPI with Swagger docs
- **Interactive UI**: Streamlit interface with examples
- **Evaluation framework**: Metrics on 15 validation cases (100% Hit@3)
