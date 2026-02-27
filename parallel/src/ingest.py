"""Ingestion script — parses CoCoA PDF and builds the vector index.

Usage: uv run python -m src.ingest
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .pdf_parser import parse_and_save
from .retrieval.semantic import build_index


def main():
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY in .env")
        sys.exit(1)

    if not Path("CoCoA.pdf").exists():
        print("ERROR: CoCoA.pdf not found")
        sys.exit(1)

    print("=" * 50)
    print("STEP 1: Parsing CoCoA PDF")
    print("=" * 50)
    parse_and_save("CoCoA.pdf", "data/chunks.json")

    print("\n" + "=" * 50)
    print("STEP 2: Building vector index")
    print("=" * 50)
    client = OpenAI(api_key=api_key)
    retriever = build_index(client, "data/chunks.json", "data/chroma_db")

    print("\n" + "=" * 50)
    print(f"DONE — {retriever.count()} entries indexed")
    print("=" * 50)


if __name__ == "__main__":
    main()
