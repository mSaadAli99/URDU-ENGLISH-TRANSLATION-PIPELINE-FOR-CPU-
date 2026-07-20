# ============================================================
# align.py — Match a hypothesis window to ground-truth text
# ============================================================

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from normalize import (
    URDU_SECTION_HEADERS,
    Language,
    extract_english_section_headers,
    normalize_for_scoring,
)

SEGMENT_LIST_KEYS = ("segments", "verified_segments", "translated_segments")
URDU_SEGMENT_TEXT_KEYS = ("text", "verified_text")
ENGLISH_SEGMENT_TEXT_KEYS = ("english_text", "verified_english", "text")


@dataclass
class AlignmentResult:
    """Aligned spans ready for WER/CER scoring."""

    hypothesis: str
    reference: str
    reference_start_word: int
    reference_end_word: int
    anchor_words: list[str]
    match_score: float
    match_ratio: float
    method: str


@dataclass
class DurationExtraction:
    """Text extracted from pipeline JSON up to a time limit."""

    text: str
    max_seconds: float
    segment_count: int
    last_end_seconds: float


def _subsequence_match_score(anchor: list[str], window: list[str]) -> int:
    """Count how many anchor words appear in order within window."""
    if not anchor:
        return 0

    score = 0
    idx = 0
    for word in anchor:
        while idx < len(window):
            if _words_similar(word, window[idx]):
                score += 1
                idx += 1
                break
            idx += 1
    return score


def _is_urdu_script_word(word: str) -> bool:
    return any("\u0600" <= ch <= "\u06ff" for ch in word)


def _build_anchor_words(
    words: list[str],
    limit: int,
    *,
    language: Language,
) -> list[str]:
    """Pick anchor words, preferring Urdu-script tokens for Urdu alignment."""
    meaningful = [w.strip() for w in words if len(w.strip()) >= 2]
    if language == "urdu":
        urdu_words = [w for w in meaningful if _is_urdu_script_word(w)]
        if len(urdu_words) >= max(4, limit // 2):
            return urdu_words[:limit]
    return _meaningful_words(meaningful, limit)


def _words_similar(a: str, b: str) -> bool:
    a = a.strip().lower()
    b = b.strip().lower()
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True

    # Common ASR/name variants in this interview corpus.
    name_aliases = {
        "majda": {"majida", "maajida", "majidah", "mijda"},
        "majida": {"majda", "maajida", "majidah"},
        "kazmi": {"kazmi", "kazme"},
        "zainab": {"zainab", "zeenab"},
        "assalam": {"assalam", "asalam", "salam", "السلام"},
        "alaikum": {"alaikum", "alikum", "alaykum", "علیکم"},
    }
    for canonical, aliases in name_aliases.items():
        if a in aliases and b in aliases:
            return True
        if a == canonical and b in aliases:
            return True
        if b == canonical and a in aliases:
            return True

    # Tolerate small ASR typos on Latin tokens.
    if re.fullmatch(r"[a-z0-9]+", a) and re.fullmatch(r"[a-z0-9]+", b):
        if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
            return True
    return False


def _meaningful_words(words: list[str], limit: int) -> list[str]:
    picked: list[str] = []
    for word in words:
        token = word.strip()
        if len(token) < 2:
            continue
        picked.append(token)
        if len(picked) >= limit:
            break
    return picked


def find_reference_span(
    hypothesis: str,
    reference: str,
    *,
    language: Language = "english",
    anchor_words: int = 10,
    min_match_ratio: float = 0.35,
) -> AlignmentResult:
    """
    Locate the best-matching span in reference for a hypothesis snippet.

    Uses ordered word anchoring: the first N meaningful hypothesis words are
    searched for inside the full reference text.
    """
    hyp_words = hypothesis.split()
    ref_words = reference.split()

    if not hyp_words:
        raise ValueError("Hypothesis text is empty.")
    if not ref_words:
        raise ValueError("Reference text is empty.")

    anchor = _build_anchor_words(hyp_words, anchor_words, language=language)
    if not anchor:
        anchor = _meaningful_words(hyp_words, anchor_words)
    if not anchor:
        raise ValueError("Could not build anchor words from hypothesis.")

    best_start = 0
    best_score = -1
    search_window = max(len(anchor) * 4, 40)

    for start in range(len(ref_words)):
        window = ref_words[start : start + search_window]
        score = _subsequence_match_score(anchor, window)
        if score > best_score:
            best_score = score
            best_start = start

    match_ratio = best_score / len(anchor)
    if match_ratio < min_match_ratio:
        raise ValueError(
            "Could not align hypothesis to ground truth. "
            f"Anchor match {best_score}/{len(anchor)} "
            f"({match_ratio:.0%}) is below threshold {min_match_ratio:.0%}. "
            f"Anchor tried: {' '.join(anchor[:6])}..."
        )

    # Reference span length follows hypothesis length, with a small margin.
    hyp_len = len(hyp_words)
    slack = max(20, int(hyp_len * 0.15))
    end = min(len(ref_words), best_start + hyp_len + slack)

    tail_anchor = _meaningful_words(hyp_words[-max(anchor_words, 15) :], 5)
    if tail_anchor:
        tail_score = _subsequence_match_score(
            tail_anchor,
            ref_words[best_start:end],
        )
        if tail_score >= max(2, len(tail_anchor) // 2):
            for end_candidate in range(best_start + hyp_len, len(ref_words) + 1):
                tail_score = _subsequence_match_score(
                    tail_anchor,
                    ref_words[best_start:end_candidate],
                )
                if tail_score >= len(tail_anchor) // 2:
                    end = end_candidate
                    break

    end = max(end, min(len(ref_words), best_start + max(hyp_len // 2, 10)))

    return AlignmentResult(
        hypothesis=hypothesis,
        reference=" ".join(ref_words[best_start:end]),
        reference_start_word=best_start,
        reference_end_word=end,
        anchor_words=anchor,
        match_score=float(best_score),
        match_ratio=match_ratio,
        method="anchor_word_search",
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick_segments(data: dict) -> list[dict]:
    for key in SEGMENT_LIST_KEYS:
        segments = data.get(key)
        if isinstance(segments, list) and segments:
            return segments
    return []


def _segment_text(segment: dict, text_keys: tuple[str, ...]) -> str:
    for key in text_keys:
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_text_by_duration(
    path: Path,
    max_seconds: float,
    *,
    language: Language,
) -> DurationExtraction:
    """
    Extract concatenated transcript text from pipeline JSON up to max_seconds.

    Includes segments whose start timestamp is strictly before the limit.
    """
    if path.suffix.lower() != ".json":
        raise ValueError(
            f"Duration filtering requires pipeline JSON with timestamps: {path}"
        )

    data = _load_json(path)
    segments = _pick_segments(data)
    if not segments:
        raise ValueError(f"No timestamped segments found in {path}")

    text_keys = (
        URDU_SEGMENT_TEXT_KEYS if language == "urdu" else ENGLISH_SEGMENT_TEXT_KEYS
    )

    selected: list[str] = []
    last_end = 0.0
    for segment in segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if start >= max_seconds:
            continue
        text = _segment_text(segment, text_keys)
        if text:
            selected.append(text)
            last_end = max(last_end, end)

    if not selected:
        raise ValueError(
            f"No segments found before {max_seconds}s in {path}"
        )

    return DurationExtraction(
        text=" ".join(selected),
        max_seconds=max_seconds,
        segment_count=len(selected),
        last_end_seconds=last_end,
    )


def extract_section_by_name(
    urdu_raw: str,
    english_raw: str,
    section_name: str,
    language: Language,
) -> str:
    """Extract a named section from Urdu or English ground truth."""
    english_headers = extract_english_section_headers(english_raw)
    if section_name not in english_headers:
        available = ", ".join(english_headers)
        raise ValueError(
            f"Section {section_name!r} not found. Available: {available}"
        )

    section_index = english_headers.index(section_name)
    urdu_header = URDU_SECTION_HEADERS[section_index]
    next_urdu_header = (
        URDU_SECTION_HEADERS[section_index + 1]
        if section_index + 1 < len(URDU_SECTION_HEADERS)
        else None
    )
    next_english_header = (
        english_headers[section_index + 1]
        if section_index + 1 < len(english_headers)
        else None
    )

    if language == "urdu":
        start = urdu_raw.find(urdu_header)
        if start == -1:
            raise ValueError(f"Urdu header not found: {urdu_header}")
        content_start = start + len(urdu_header)
        if next_urdu_header:
            end = urdu_raw.find(next_urdu_header, content_start)
            chunk = urdu_raw[content_start:end if end != -1 else None]
        else:
            chunk = urdu_raw[content_start:]
        return chunk.strip()

    # English: header on its own line.
    pattern = re.escape(section_name)
    match = re.search(rf"^{pattern}\s*$", english_raw, re.MULTILINE)
    if not match:
        raise ValueError(f"English section header not found: {section_name}")
    content_start = match.end()
    if next_english_header:
        next_match = re.search(
            rf"^{re.escape(next_english_header)}\s*$",
            english_raw[content_start:],
            re.MULTILINE,
        )
        end = content_start + next_match.start() if next_match else len(english_raw)
        chunk = english_raw[content_start:end]
    else:
        chunk = english_raw[content_start:]
    return chunk.strip()


def align_hypothesis_to_reference(
    hypothesis_raw: str,
    reference_raw: str,
    language: Language,
    *,
    is_reference: bool = True,
) -> tuple[AlignmentResult, str, str]:
    """
    Normalize texts, align hypothesis inside reference, return aligned pair.

    Returns (alignment, cleaned_hypothesis_full, cleaned_reference_full).
    """
    hyp_norm = normalize_for_scoring(hypothesis_raw, language, is_reference=False)
    ref_norm = normalize_for_scoring(reference_raw, language, is_reference=is_reference)

    alignment = find_reference_span(hyp_norm.clean, ref_norm.clean, language=language)
    return alignment, hyp_norm.clean, ref_norm.clean
