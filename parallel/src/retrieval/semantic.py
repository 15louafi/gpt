"""Semantic retriever backed by ChromaDB + OpenAI embeddings."""

import json
import time

import chromadb
from openai import OpenAI

from src.models import SearchResult
from src.pdf_parser import CodeChunk

EMBEDDING_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100
CHROMA_UPSERT_BATCH = 500


class SemanticRetriever:
    def __init__(self, openai_client: OpenAI, persist_dir: str = "data/chroma_db"):
        self._client = openai_client
        self._chroma = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._chroma.get_or_create_collection(
            name="cim10_codes",
            metadata={"hnsw:space": "cosine"},
        )

    # -- Retriever interface ---------------------------------------------------

    def search(self, query: str, n_results: int = 15) -> list[SearchResult]:
        embedding = self._embed([query])[0]
        raw = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, self.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if not raw or not raw["documents"]:
            return []

        results = [
            SearchResult(
                document=doc,
                code=meta["code"],
                label=meta["label"],
                chapter=meta["chapter"],
                page_number=meta["page_number"],
                similarity=1 - dist,
            )
            for doc, meta, dist in zip(
                raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
            )
        ]
        results.sort(key=lambda r: r.similarity, reverse=True)
        return results

    # -- Indexing ---------------------------------------------------------------

    def index_chunks(self, chunks: list[CodeChunk]) -> None:
        texts = [c.to_embedding_text() for c in chunks]
        ids = [f"{c.code}_{c.page_number}_{i}" for i, c in enumerate(chunks)]
        metadatas = [
            {
                "code": c.code,
                "label": c.label,
                "chapter": c.chapter,
                "chapter_title": c.chapter_title,
                "page_number": c.page_number,
            }
            for c in chunks
        ]

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self._embed(texts)

        for i in range(0, len(chunks), CHROMA_UPSERT_BATCH):
            end = min(i + CHROMA_UPSERT_BATCH, len(chunks))
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
        print(f"Indexed {len(chunks)} chunks")

    def count(self) -> int:
        return self._collection.count()

    # -- Embedding helper -------------------------------------------------------

    def _embed(self, texts: list[str]) -> list[list[float]]:
        all_emb: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            resp = self._client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
            all_emb.extend(item.embedding for item in resp.data)
            if i + EMBED_BATCH_SIZE < len(texts):
                time.sleep(0.1)
        return all_emb


def build_index(
    openai_client: OpenAI,
    chunks_path: str = "data/chunks.json",
    persist_dir: str = "data/chroma_db",
) -> SemanticRetriever:
    """Parse saved chunks JSON and build the vector index."""
    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    chunks = [CodeChunk(**item) for item in data]
    print(f"Loaded {len(chunks)} chunks from {chunks_path}")

    retriever = SemanticRetriever(openai_client, persist_dir)
    retriever.index_chunks(chunks)
    print(f"Index built: {retriever.count()} entries")
    return retriever
