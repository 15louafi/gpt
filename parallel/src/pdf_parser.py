"""
CoCoA PDF Parser - Extracts CIM-10 code entries from the CoCoA PDF.

Uses PyMuPDF to extract text, then identifies CIM-10 codes appearing on their own line
(the PDF layout puts P/R/A indicators, then the code, then the label on separate lines).
Each code gets a chunk with its surrounding context for embedding.
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf

# CIM-10 code alone on a line: letter + 2 digits, optionally .digit(s), +/-
CODE_PATTERN = re.compile(r"^([A-Z]\d{2}(?:\.\d{1,2})?(?:[+\-]\d?)?)(?:\s*[†*])?$")
# Noise lines to skip: P/R/A indicators, SSR markers, bare numbers, page footers
SKIP_LINE = re.compile(r"^(P|R|A|SSR|\d{1,2}|2023\s*[–\-].*)$")

CHAPTER_MAP = {
    "I": ("A00-B99", "Certaines maladies infectieuses et parasitaires"),
    "II": ("C00-D48", "Tumeurs"),
    "III": ("D50-D89", "Maladies du sang et des organes hématopoïétiques"),
    "IV": ("E00-E90", "Maladies endocriniennes, nutritionnelles et métaboliques"),
    "V": ("F00-F99", "Troubles mentaux et du comportement"),
    "VI": ("G00-G99", "Maladies du système nerveux"),
    "VII": ("H00-H59", "Maladies de l'œil et de ses annexes"),
    "VIII": ("H60-H95", "Maladies de l'oreille et de l'apophyse mastoïde"),
    "IX": ("I00-I99", "Maladies de l'appareil circulatoire"),
    "X": ("J00-J99", "Maladies de l'appareil respiratoire"),
    "XI": ("K00-K93", "Maladies de l'appareil digestif"),
    "XII": ("L00-L99", "Maladies de la peau et du tissu cellulaire sous-cutané"),
    "XIII": (
        "M00-M99",
        "Maladies du système ostéo-articulaire, des muscles et du tissu conjonctif",
    ),
    "XIV": ("N00-N99", "Maladies de l'appareil génito-urinaire"),
    "XV": ("O00-O99", "Grossesse, accouchement et puerpéralité"),
    "XVI": ("P00-P96", "Affections dont l'origine se situe dans la période périnatale"),
    "XVII": ("Q00-Q99", "Malformations congénitales et anomalies chromosomiques"),
    "XVIII": ("R00-R99", "Symptômes, signes et résultats anormaux, non classés ailleurs"),
    "XIX": ("S00-T98", "Lésions traumatiques, empoisonnements"),
    "XX": ("V01-Y98", "Causes externes de morbidité et de mortalité"),
    "XXI": ("Z00-Z99", "Facteurs influant sur l'état de santé"),
    "XXII": ("U00-U99", "Codes d'utilisation particulière"),
}


@dataclass
class CodeChunk:
    """A CIM-10 code entry with its context from CoCoA."""

    code: str
    label: str
    chapter: str
    chapter_title: str
    full_text: str  # The code + all surrounding context lines
    page_number: int = 0

    def to_embedding_text(self) -> str:
        """Text representation used for embedding."""
        parts = [f"Code CIM-10: {self.code}", f"Libellé: {self.label}"]
        if self.chapter_title:
            parts.append(f"Chapitre: {self.chapter_title}")
        parts.append(self.full_text)
        return "\n".join(parts)


def get_chapter_for_code(code: str) -> tuple[str, str]:
    """Return (chapter_number, chapter_title) for a CIM-10 code."""
    if len(code) < 3:
        return "", ""
    letter, num = code[0], int(code[1:3])
    for chap, (code_range, title) in CHAPTER_MAP.items():
        s, e = code_range.split("-")
        s_ord = ord(s[0]) * 100 + int(s[1:3])
        e_ord = ord(e[0]) * 100 + int(e[1:3])
        c_ord = ord(letter) * 100 + num
        if s_ord <= c_ord <= e_ord:
            return chap, title
    return "", ""


def clean_lines(text: str) -> list[str]:
    """Filter out noise lines from PDF-extracted text."""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        if not s:
            continue
        if SKIP_LINE.match(s):
            continue
        if s.startswith("CHAPITRE") and ":" in s:
            continue
        out.append(s)
    return out


def parse_pdf(pdf_path: str) -> list[CodeChunk]:
    """Parse the CoCoA PDF into code chunks."""
    doc = pymupdf.open(pdf_path)
    chunks: list[CodeChunk] = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        lines = clean_lines(doc[page_idx].get_text("text"))

        i = 0
        while i < len(lines):
            m = CODE_PATTERN.match(lines[i])
            if not m:
                i += 1
                continue

            code = m.group(1)

            # Next non-code line is the label
            label = ""
            if i + 1 < len(lines) and not CODE_PATTERN.match(lines[i + 1]):
                label = lines[i + 1]

            if not label or len(label) < 2 or (label.startswith("20") and len(label) < 15):
                i += 1
                continue

            # Collect all context lines until the next code
            j = i + 2
            context = [f"{code}  {label}"]
            while j < len(lines):
                if CODE_PATTERN.match(lines[j]):
                    break
                context.append(lines[j])
                j += 1

            chap, chap_title = get_chapter_for_code(code)

            chunks.append(
                CodeChunk(
                    code=code,
                    label=label,
                    chapter=chap,
                    chapter_title=chap_title,
                    full_text="\n".join(context),
                    page_number=page_num,
                )
            )
            i = j

    doc.close()
    return chunks


def parse_and_save(pdf_path: str, output_path: str) -> list[CodeChunk]:
    """Parse CoCoA PDF and save chunks to JSON."""
    print(f"Parsing {pdf_path}...")
    chunks = parse_pdf(pdf_path)
    print(f"Extracted {len(chunks)} code entries")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in chunks], f, ensure_ascii=False, indent=2)

    print(f"Saved to {output_path}")
    return chunks


if __name__ == "__main__":
    chunks = parse_and_save("CoCoA.pdf", "data/chunks.json")
    for c in chunks[:5]:
        print(f"\n{c.code}: {c.label} (ch.{c.chapter}, p.{c.page_number})")
