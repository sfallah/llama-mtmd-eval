"""OCR scoring — the single source of truth for CER/chrF, so llama.cpp and HF
outputs are scored identically.

Ported verbatim from the llama.cpp in-repo test
(tools/mtmd/tests/test-deepseek-ocr.py): NFC-normalize + whitespace-collapse,
fuzzy local alignment of the OCR span against the ground truth, then CER (jiwer)
and chrF (sacrebleu). Third-party libs are imported lazily so importing this
module (and the CLI) stays cheap and torch-free.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# Matches <|ref|>..<|/ref|> and <|det|>..<|/det|> grounding spans (Unlimited-OCR).
# The backreference ties the close tag to the open tag; DOTALL so it spans newlines.
GROUNDING_TAG_RE = re.compile(r"<\|(ref|det)\|>.*?<\|/\1\|>", re.DOTALL)


@dataclass(frozen=True)
class Score:
    cer: float
    chrf: float
    aligned: str = ""   # the OCR span that was scored (normalized)


def strip_grounding(text: str) -> str:
    """Drop <|ref|>..<|/ref|> / <|det|>..<|/det|> markup, matching the cleaned
    result.md the HF reference scores against."""
    return GROUNDING_TAG_RE.sub("", text)


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse whitespace, so line-wrap and spacing don't
    count as CER errors."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def locally_align(expected: str, ocr_out: str) -> str:
    """Return the span of `ocr_out` that best matches `expected`.

    The ground truth covers part of the article body while the OCR output may
    cover the whole page; fuzzy partial-ratio matching picks out the body span so
    unrelated text doesn't disturb CER / chrF.
    """
    from rapidfuzz import fuzz
    alignment = fuzz.partial_ratio_alignment(expected, ocr_out)
    if alignment is None or alignment.dest_end <= alignment.dest_start:
        return ocr_out
    return ocr_out[alignment.dest_start:alignment.dest_end]


def compute_cer(expected: str, ocr_out: str) -> float:
    """Character Error Rate (lower is better; 0 = perfect)."""
    import jiwer
    return jiwer.cer(expected, ocr_out)


def compute_chrf(expected: str, ocr_out: str) -> float:
    """chrF on 0-100 (higher is better)."""
    from sacrebleu.metrics import CHRF
    return CHRF().sentence_score(ocr_out, [expected]).score


def score(raw_ocr: str, ground_truth: str, *, strip_grounding_markup: bool = False) -> Score:
    """Full pipeline: optional grounding-strip -> normalize both -> align the OCR
    span -> CER + chrF. Used identically for llama and HF outputs."""
    if strip_grounding_markup:
        raw_ocr = strip_grounding(raw_ocr)
    expected = normalize_text(ground_truth)
    ocr = normalize_text(raw_ocr)
    aligned = locally_align(expected, ocr)
    return Score(cer=compute_cer(expected, aligned), chrf=compute_chrf(expected, aligned),
                 aligned=aligned)
