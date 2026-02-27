"""Streamlit UI for testing the CIM-10 RAG system.

Usage: uv run streamlit run src/app.py
"""

import os
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from src.generation import StructuredGenerator
from src.pipeline import Pipeline
from src.reranking import LLMReranker
from src.retrieval import ExpandedRetriever
from src.retrieval.semantic import SemanticRetriever

load_dotenv()

st.set_page_config(page_title="CIM-10 RAG - CoCoA", page_icon="🏥", layout="wide")

EXAMPLES = [
    "Dyspnée (difficulté respiratoire) à l'effort et à la parole",
    "Toux purulente",
    "Fièvre",
    "Œdème des membres inférieurs",
    "Hyponatrémie (faible taux de sodium)",
    "Hypercalcémie (taux élevé de calcium)",
    "Tachycardie",
    "Détresse respiratoire aiguë",
    "Hypertension artérielle (HTA)",
    "Diabète de type 2 non insulinodépendant",
    "Fibrillation auriculaire",
    "Infection pulmonaire à Haemophilus influenzae",
    "Insuffisance respiratoire aiguë hypoxémique sur décompensation cardiaque globale",
    "Pneumopathie à Haemophilus influenzae",
    "Insuffisance rénale aiguë fonctionnelle (secondaire à la déplétion)",
    "Acidose mixte (secondaire à la décompensation respiratoire)",
]


@st.cache_resource
def init_pipeline():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key)
    semantic = SemanticRetriever(client)
    if semantic.count() == 0:
        return None
    return Pipeline(
        retriever=ExpandedRetriever(semantic, client),
        reranker=LLMReranker(client),
        generator=StructuredGenerator(client),
    )


def main():
    st.title("🏥 CIM-10 RAG System - CoCoA")
    st.markdown("Suggestion de codes CIM-10 basée sur le document **CoCoA** via RAG + GPT-4.")

    pipeline = init_pipeline()
    if pipeline is None:
        st.error(
            "Pipeline non initialisé. Vérifiez `.env` et lancez `uv run python -m src.ingest`."
        )
        return

    st.sidebar.markdown("**LLM:** GPT-4.1 | **Embeddings:** text-embedding-3-small")
    st.sidebar.markdown("---")
    selected = st.sidebar.selectbox(
        "Exemples:",
        [""] + EXAMPLES,
        format_func=lambda x: x[:55] + "..." if len(x) > 55 else x if x else "-- Sélectionner --",
    )

    query = st.text_area(
        "Diagnostic / Symptôme / Maladie", value=selected, height=80, max_chars=200
    )
    submit = st.button("🔍 Rechercher", type="primary")

    if submit and query.strip():
        with st.spinner("Recherche en cours..."):
            start = time.time()
            result = pipeline.query(query.strip())
            elapsed = time.time() - start

        st.success(f"Résultats en {elapsed:.1f}s")

        for i, code in enumerate(result.codes):
            badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(code.confidence, "⚪")
            with st.expander(f"{badge} **{code.code}** — {code.label}", expanded=i < 3):
                st.markdown(f"**Pertinence:** {code.relevance}")
                if code.cocoa_info:
                    st.markdown(f"**CoCoA:** {code.cocoa_info}")

        if result.coding_rules:
            st.header("📋 Règles de codage")
            for r in result.coding_rules:
                st.markdown(f"- {r}")

        if result.warnings:
            for w in result.warnings:
                st.warning(w)

        with st.expander("📈 Métadonnées"):
            st.json(result.model_dump())


if __name__ == "__main__":
    main()
