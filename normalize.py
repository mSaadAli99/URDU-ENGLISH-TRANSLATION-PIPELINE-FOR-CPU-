# ============================================================
# normalize.py — Text normalization for ASR/MT evaluation
# ============================================================

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

Language = Literal["urdu", "english"]

# Known speaker labels in ground-truth and pipeline output.
SPEAKER_LABELS = (
    r"Zainab Faisal",
    r"Dr\.?\s+Majida(?:\s+Kazmi)?",
    r"زینب فیصل",
    r"ڈاکٹر\s*ماجدہ(?:\s*کاظمی)?",
)

SPEAKER_LABEL_RE = re.compile(
    rf"(?:^|\s)(?:{'|'.join(SPEAKER_LABELS)})\s*:?\s*",
    re.IGNORECASE,
)

SPEAKER_LABEL_INLINE_RE = re.compile(
    rf"(?:{'|'.join(SPEAKER_LABELS)})\s*:?\s*",
    re.IGNORECASE,
)

# Urdu section headers aligned with English_translation.txt sections.
URDU_SECTION_HEADERS = (
    "تعارف اور آغاز",
    "کام اور زندگی کا توازن",
    "تعلیمی انتخاب اور مقصد (Intent)",
    "کیریئر کے چیلنجز اور خاندانی تعاون",
    "ذاتی احساسِ جرم (Self-guilt) اور معاشرتی رویے",
    "این ای ڈی اسٹیم (STEM) سینٹر اور اساتذہ کی تربیت",
    "مینٹرشپ اور نئی نسل (Gen Z)",
    "لیب کے تجربات اور طلباء کی حوصلہ افزائی",
    "انجینئرنگ ایکسیلنس ایوارڈز",
    "مستقبل کے منصوبے: ڈیجیٹل سسٹین ایبلٹی",
    "مصنوعی ذہانت (AI) اور ناکامی (Failure)",
    "خاتمہ",
)

# Pipeline artifacts that should not affect scoring.
PIPELINE_NOISE_RE = re.compile(r"\[LOW CONFIDENCE\]", re.IGNORECASE)

# Arabic → Urdu character unification fallback (when urduhack is unavailable).
ARABIC_TO_URDU_MAP = str.maketrans({
    "\u064A": "\u06CC",  # ي → ی
    "\u0643": "\u06A9",  # ك → ک
    "\u0649": "\u06CC",  # ى → ی
    "\u0629": "\u06C1",  # ة → ہ
    "\u0623": "\u0627",  # أ → ا
    "\u0625": "\u0627",  # إ → ا
    "\u0622": "\u0622",  # آ
    "\u0640": "",        # tatweel
})

URDU_PUNCT_MAP = str.maketrans({
    "\u06D4": " ",   # Urdu full stop ۔ 
    "\u061F": " ",   # ؟
    "\u061B": " ",   # ؛
    "\u060C": " ",   # ،
    ".": " ",
    "?": " ",
    "!": " ",
    ",": " ",
    ";": " ",
    ":": " ",
    "(": " ",
    ")": " ",
    '"': " ",
    "'": " ",
    "—": " ",
    "-": " ",
})

ENGLISH_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass
class NormalizationResult:
    """Original and cleaned text plus a reproducibility log."""

    original: str
    clean: str
    language: Language
    section_count: int
    steps_applied: list[str] = field(default_factory=list)


def _try_urduhack_normalize(text: str) -> tuple[str, bool]:
    try:
        from urduhack.normalization import normalize as urdu_normalize

        return urdu_normalize(text), True
    except Exception:
        return text, False


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_english_section_headers(text: str) -> list[str]:
    """Return standalone section header lines from English ground truth."""
    headers: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if ":" in stripped:
            continue
        headers.append(stripped)
    return headers


def extract_urdu_section_headers(text: str) -> list[str]:
    """Return Urdu section headers found in the reference text."""
    found: list[str] = []
    for header in URDU_SECTION_HEADERS:
        if header in text:
            found.append(header)
    return found


def count_sections(text: str, language: Language) -> int:
    if language == "english":
        return len(extract_english_section_headers(text))
    return len(extract_urdu_section_headers(text))


def strip_section_headers(text: str, language: Language) -> str:
    if language == "english":
        headers = extract_english_section_headers(text)
    else:
        headers = list(URDU_SECTION_HEADERS)

    for header in sorted(headers, key=len, reverse=True):
        text = text.replace(header, " ")
    return text


def strip_speaker_labels(text: str) -> str:
    text = SPEAKER_LABEL_INLINE_RE.sub(" ", text)
    text = re.sub(r"زینب\s+فیصل\s+نے", " ", text)
    return text


def strip_pipeline_noise(text: str) -> str:
    return PIPELINE_NOISE_RE.sub(" ", text)


def normalize_urdu_characters(text: str) -> tuple[str, list[str]]:
    steps: list[str] = []
    text = normalize_unicode(text)
    text = text.translate(ARABIC_TO_URDU_MAP)

    normalized, used_urduhack = _try_urduhack_normalize(text)
    text = normalized
    if used_urduhack:
        steps.append("urduhack.normalize (diacritics, Arabic→Urdu forms, spacing)")
    else:
        steps.append("manual Arabic→Urdu character mapping (fallback)")

    return text, steps


def normalize_urdu_punctuation(text: str) -> str:
    text = text.translate(URDU_PUNCT_MAP)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_english_text(text: str) -> str:
    text = normalize_unicode(text)
    text = text.lower()
    text = ENGLISH_PUNCT_RE.sub(" ", text)
    return collapse_whitespace(text)


def normalize_for_scoring(
    text: str,
    language: Language,
    *,
    is_reference: bool = False,
) -> NormalizationResult:
    """
    Normalize text for WER/CER comparison.

    Reference ground-truth files have section headers and speaker labels removed.
    Hypothesis (pipeline) files only have pipeline noise stripped.
    """
    original = text
    steps: list[str] = ["unicode NFC normalization"]
    working = normalize_unicode(text)
    section_count = count_sections(working, language)

    if is_reference:
        working = strip_section_headers(working, language)
        steps.append("strip section headers")
        working = strip_speaker_labels(working)
        steps.append("strip speaker labels")

    working = strip_pipeline_noise(working)
    steps.append("strip pipeline noise markers")

    working = collapse_whitespace(working)
    steps.append("collapse whitespace")

    if language == "urdu":
        working, urdu_steps = normalize_urdu_characters(working)
        steps.extend(urdu_steps)
        working = normalize_urdu_punctuation(working)
        steps.append("normalize Urdu punctuation (۔ vs ., etc.)")
        working = collapse_whitespace(working)
        steps.append("remove trailing punctuation variance")
    else:
        working = normalize_english_text(working)
        steps.append("lowercase English")
        steps.append("strip English punctuation")

    return NormalizationResult(
        original=original,
        clean=working,
        language=language,
        section_count=section_count,
        steps_applied=steps,
    )


def get_normalization_log(language: Language) -> list[str]:
    """Return the canonical normalization pipeline for a language."""
    result = normalize_for_scoring("", language, is_reference=True)
    return result.steps_applied
