# ============================================================
# metrics.py — WER / CER computation via jiwer
# ============================================================

from __future__ import annotations

from dataclasses import dataclass

import jiwer


@dataclass
class MetricResult:
    wer: float
    cer: float
    reference_length_words: int
    reference_length_chars: int
    hypothesis_length_words: int
    hypothesis_length_chars: int


def _word_count(text: str) -> int:
    return len(text.split()) if text else 0


def _char_count(text: str) -> int:
    return len(text.replace(" ", "")) if text else 0


def compute_wer_cer(reference: str, hypothesis: str) -> MetricResult:
    """
    Compute word and character error rates between reference and hypothesis.

    Returns raw ratios in [0, inf); callers format as percentages for display.
    """
    reference = reference or ""
    hypothesis = hypothesis or ""

    if not reference.strip():
        raise ValueError("Reference text is empty after normalization.")

    if not hypothesis.strip():
        raise ValueError("Hypothesis text is empty after normalization.")

    wer = float(jiwer.wer(reference, hypothesis))
    cer = float(jiwer.cer(reference, hypothesis))

    return MetricResult(
        wer=wer,
        cer=cer,
        reference_length_words=_word_count(reference),
        reference_length_chars=_char_count(reference),
        hypothesis_length_words=_word_count(hypothesis),
        hypothesis_length_chars=_char_count(hypothesis),
    )


def format_percentage(ratio: float) -> str:
    return f"{ratio * 100:.2f}%"
