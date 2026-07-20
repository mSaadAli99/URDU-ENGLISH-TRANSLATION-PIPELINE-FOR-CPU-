#!/usr/bin/env python3
# ============================================================
# evaluate.py — CLI for Urdu ASR + translation evaluation
#
# Full-document evaluation:
#   python evaluate.py \
#     --urdu-hypothesis outputs/1_urdu_transcripts/test_audio_urdu_transcript.json \
#     --english-hypothesis outputs/3_english_translations/test_audio_english_translation.json
#
# First 5 minutes only (align hypothesis words inside ground truth):
#   python evaluate.py \
#     --urdu-hypothesis outputs/1_urdu_transcripts/test_audio_urdu_transcript.json \
#     --english-hypothesis outputs/3_english_translations/test_audio_english_translation.json \
#     --max-minutes 5
#
# One named section only:
#   python evaluate.py ... --section "Introduction and Beginning"
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import config
from align import (
    AlignmentResult,
    DurationExtraction,
    extract_section_by_name,
    extract_text_by_duration,
    find_reference_span,
)
from metrics import compute_wer_cer, format_percentage
from normalize import (
    NormalizationResult,
    count_sections,
    get_normalization_log,
    normalize_for_scoring,
)

DEFAULT_URDU_REFERENCE = os.path.join(
    config.BASE_DIR, "notebooks", "Groundtruth", "Urdu_Transcription.txt"
)
DEFAULT_ENGLISH_REFERENCE = os.path.join(
    config.BASE_DIR, "notebooks", "Groundtruth", "English_translation.txt"
)
DEFAULT_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "evaluation")

URDU_JSON_KEYS = (
    "full_urdu_text",
    "verified_full_text",
    "original_full_text",
    "urdu_full_text",
)
ENGLISH_JSON_KEYS = (
    "english_full_text",
    "verified_english_full",
    "english_translation",
)


class EvaluationError(Exception):
    """Raised when evaluation inputs are invalid."""


def _read_text_file(path: Path) -> str:
    if not path.exists():
        raise EvaluationError(f"File not found: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise EvaluationError(f"File is empty: {path}")
    return text


def _extract_from_json(data: dict, keys: tuple[str, ...], path: Path) -> str:
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise EvaluationError(
        f"Could not find hypothesis text in {path}. "
        f"Tried keys: {', '.join(keys)}"
    )


def load_text(path: str | Path, *, json_keys: tuple[str, ...] | None = None) -> tuple[str, Path]:
    path = Path(path)
    if not path.exists():
        raise EvaluationError(f"File not found: {path}")

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"Invalid JSON in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise EvaluationError(f"Expected JSON object in {path}")

        keys = json_keys
        if keys is None:
            if any(k in data for k in URDU_JSON_KEYS):
                keys = URDU_JSON_KEYS
            elif any(k in data for k in ENGLISH_JSON_KEYS):
                keys = ENGLISH_JSON_KEYS
            else:
                raise EvaluationError(
                    f"No recognized text field in {path}. "
                    f"Expected one of: {', '.join(URDU_JSON_KEYS + ENGLISH_JSON_KEYS)}"
                )
        return _extract_from_json(data, keys, path), path

    text = _read_text_file(path)
    return text, path


def validate_section_alignment(
    urdu_reference: NormalizationResult,
    english_reference: NormalizationResult,
) -> None:
    urdu_sections = urdu_reference.section_count
    english_sections = english_reference.section_count
    if urdu_sections != english_sections:
        raise EvaluationError(
            "Section count mismatch between Urdu and English ground truth "
            f"({urdu_sections} Urdu vs {english_sections} English). "
            "Check that both reference files use the same section headers."
        )


def resolve_duration_limit(args: argparse.Namespace) -> float | None:
    if args.max_seconds is not None and args.max_minutes is not None:
        raise EvaluationError("Use only one of --max-minutes or --max-seconds.")
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            raise EvaluationError("--max-seconds must be positive.")
        return float(args.max_seconds)
    if args.max_minutes is not None:
        if args.max_minutes <= 0:
            raise EvaluationError("--max-minutes must be positive.")
        return float(args.max_minutes) * 60.0
    return None


def load_hypothesis_text(
    path: Path,
    language: str,
    max_seconds: float | None,
) -> tuple[str, DurationExtraction | None]:
    if max_seconds is not None:
        if path.suffix.lower() != ".json":
            raise EvaluationError(
                f"--max-minutes/--max-seconds requires a pipeline JSON file, got: {path}"
            )
        extraction = extract_text_by_duration(path, max_seconds, language=language)
        return extraction.text, extraction
    text, _ = load_text(path)
    return text, None


def prepare_comparison_pair(
    hypothesis_raw: str,
    reference_raw: str,
    language: str,
    *,
    align: bool,
    section_name: str | None,
    english_reference_raw: str | None = None,
) -> tuple[str, str, AlignmentResult | None, NormalizationResult, NormalizationResult]:
    if section_name:
        if language == "urdu":
            reference_raw = extract_section_by_name(
                reference_raw,
                english_reference_raw or "",
                section_name,
                "urdu",
            )
        else:
            reference_raw = extract_section_by_name(
                reference_raw,
                reference_raw,
                section_name,
                "english",
            )

    hyp_norm = normalize_for_scoring(hypothesis_raw, language, is_reference=False)
    ref_norm = normalize_for_scoring(reference_raw, language, is_reference=True)

    alignment = None
    ref_for_scoring = ref_norm.clean
    hyp_for_scoring = hyp_norm.clean

    if align:
        alignment = find_reference_span(hyp_norm.clean, ref_norm.clean, language=language)
        ref_for_scoring = alignment.reference
        hyp_for_scoring = alignment.hypothesis

    return hyp_for_scoring, ref_for_scoring, alignment, hyp_norm, ref_norm


def save_artifacts(
    output_dir: Path,
    urdu_reference: NormalizationResult,
    english_reference: NormalizationResult,
    urdu_hypothesis: NormalizationResult,
    english_hypothesis: NormalizationResult,
    *,
    urdu_reference_aligned: str | None = None,
    english_reference_aligned: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("urdu_reference_original.txt", urdu_reference.original),
        ("urdu_reference_clean.txt", urdu_reference.clean),
        ("english_reference_original.txt", english_reference.original),
        ("english_reference_clean.txt", english_reference.clean),
        ("urdu_hypothesis_clean.txt", urdu_hypothesis.clean),
        ("english_hypothesis_clean.txt", english_hypothesis.clean),
    ]
    if urdu_reference_aligned is not None:
        pairs.append(("urdu_reference_aligned.txt", urdu_reference_aligned))
    if english_reference_aligned is not None:
        pairs.append(("english_reference_aligned.txt", english_reference_aligned))

    for filename, content in pairs:
        (output_dir / filename).write_text(content, encoding="utf-8")


def print_results_table(
    urdu_metrics,
    english_metrics,
    urdu_hypothesis_path: Path,
    urdu_reference_path: Path,
    english_hypothesis_path: Path,
    english_reference_path: Path,
    *,
    mode: str,
    duration_info: DurationExtraction | None = None,
    urdu_alignment: AlignmentResult | None = None,
    english_alignment: AlignmentResult | None = None,
    section_name: str | None = None,
) -> None:
    col_w = 22
    print()
    print("=" * 72)
    print("  URDU PIPELINE EVALUATION RESULTS")
    print("=" * 72)
    print(f"  Mode              : {mode}")
    if duration_info:
        print(
            f"  Duration window   : first {duration_info.max_seconds / 60:.1f} min "
            f"({duration_info.segment_count} segments, ends at {duration_info.last_end_seconds:.1f}s)"
        )
    if section_name:
        print(f"  Section           : {section_name}")
    print(f"  {'Metric':<{col_w}} {'Urdu (ASR)':>14} {'English (MT)':>14}")
    print("-" * 72)
    print(
        f"  {'WER':<{col_w}} "
        f"{format_percentage(urdu_metrics.wer):>14} "
        f"{format_percentage(english_metrics.wer):>14}"
    )
    print(
        f"  {'CER':<{col_w}} "
        f"{format_percentage(urdu_metrics.cer):>14} "
        f"{format_percentage(english_metrics.cer):>14}"
    )
    print("-" * 72)
    if urdu_alignment:
        print(
            f"  Urdu anchor match : {urdu_alignment.match_score:.0f}/"
            f"{len(urdu_alignment.anchor_words)} words "
            f"({urdu_alignment.match_ratio:.0%}) at ref word {urdu_alignment.reference_start_word}"
        )
    if english_alignment:
        print(
            f"  English anchor match: {english_alignment.match_score:.0f}/"
            f"{len(english_alignment.anchor_words)} words "
            f"({english_alignment.match_ratio:.0%}) at ref word {english_alignment.reference_start_word}"
        )
    print(f"  Urdu hypothesis   : {urdu_hypothesis_path}")
    print(f"  Urdu reference    : {urdu_reference_path}")
    print(f"  English hypothesis: {english_hypothesis_path}")
    print(f"  English reference : {english_reference_path}")
    print("=" * 72)
    print()


def print_normalization_log() -> None:
    print("Normalization steps applied:")
    print("  Urdu:")
    for step in get_normalization_log("urdu"):
        print(f"    - {step}")
    print("  English:")
    for step in get_normalization_log("english"):
        print(f"    - {step}")
    print()


def build_results_payload(
    urdu_metrics,
    english_metrics,
    urdu_hypothesis_path: Path,
    urdu_reference_path: Path,
    english_hypothesis_path: Path,
    english_reference_path: Path,
    *,
    mode: str,
    duration_info: DurationExtraction | None,
    urdu_alignment: AlignmentResult | None,
    english_alignment: AlignmentResult | None,
    section_name: str | None,
) -> dict:
    timestamp = datetime.now(timezone.utc).isoformat()
    payload = {
        "urdu_wer": round(urdu_metrics.wer, 6),
        "urdu_cer": round(urdu_metrics.cer, 6),
        "english_wer": round(english_metrics.wer, 6),
        "english_cer": round(english_metrics.cer, 6),
        "urdu_wer_percent": round(urdu_metrics.wer * 100, 2),
        "urdu_cer_percent": round(urdu_metrics.cer * 100, 2),
        "english_wer_percent": round(english_metrics.wer * 100, 2),
        "english_cer_percent": round(english_metrics.cer * 100, 2),
        "timestamp": timestamp,
        "hypothesis_file": str(urdu_hypothesis_path),
        "reference_file": str(urdu_reference_path),
        "urdu_hypothesis_file": str(urdu_hypothesis_path),
        "urdu_reference_file": str(urdu_reference_path),
        "english_hypothesis_file": str(english_hypothesis_path),
        "english_reference_file": str(english_reference_path),
        "evaluation_mode": mode,
        "reference_section_count": count_sections(
            urdu_reference_path.read_text(encoding="utf-8"), "urdu"
        ),
        "normalization_steps": {
            "urdu": get_normalization_log("urdu"),
            "english": get_normalization_log("english"),
        },
        "reference_lengths": {
            "urdu_words": urdu_metrics.reference_length_words,
            "urdu_chars": urdu_metrics.reference_length_chars,
            "english_words": english_metrics.reference_length_words,
            "english_chars": english_metrics.reference_length_chars,
        },
    }

    if duration_info:
        payload["duration_window"] = {
            "max_seconds": duration_info.max_seconds,
            "max_minutes": round(duration_info.max_seconds / 60.0, 3),
            "segment_count": duration_info.segment_count,
            "last_end_seconds": duration_info.last_end_seconds,
        }

    if section_name:
        payload["section"] = section_name

    if urdu_alignment:
        payload["urdu_alignment"] = {
            "method": urdu_alignment.method,
            "anchor_words": urdu_alignment.anchor_words,
            "match_score": urdu_alignment.match_score,
            "match_ratio": round(urdu_alignment.match_ratio, 4),
            "reference_start_word": urdu_alignment.reference_start_word,
            "reference_end_word": urdu_alignment.reference_end_word,
        }

    if english_alignment:
        payload["english_alignment"] = {
            "method": english_alignment.method,
            "anchor_words": english_alignment.anchor_words,
            "match_score": english_alignment.match_score,
            "match_ratio": round(english_alignment.match_ratio, 4),
            "reference_start_word": english_alignment.reference_start_word,
            "reference_end_word": english_alignment.reference_end_word,
        }

    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Urdu ASR and English translation against ground truth."
    )
    parser.add_argument(
        "--urdu-hypothesis",
        required=True,
        help="Pipeline Urdu transcript (.json or .txt)",
    )
    parser.add_argument(
        "--urdu-reference",
        default=DEFAULT_URDU_REFERENCE,
        help="Ground-truth Urdu transcript (.txt)",
    )
    parser.add_argument(
        "--english-hypothesis",
        required=True,
        help="Pipeline English translation (.json or .txt)",
    )
    parser.add_argument(
        "--english-reference",
        default=DEFAULT_ENGLISH_REFERENCE,
        help="Ground-truth English translation (.txt)",
    )
    parser.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="Evaluate only pipeline segments from the first N minutes, then align to ground truth",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="Same as --max-minutes, but in seconds",
    )
    parser.add_argument(
        "--section",
        default=None,
        help='Evaluate one ground-truth section only (English header name, e.g. "Introduction and Beginning")',
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        help="Disable anchor-word search in ground truth (compare full cleaned texts)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for results.json and cleaned text artifacts",
    )
    parser.add_argument(
        "--results-file",
        default=None,
        help="Override path for results.json (default: <output-dir>/results.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_path = Path(args.results_file) if args.results_file else output_dir / "results.json"

    try:
        max_seconds = resolve_duration_limit(args)
        align = not args.no_align
        if max_seconds is not None or args.section:
            align = align and not args.no_align
        elif args.no_align:
            align = False

        urdu_hypothesis_path = Path(args.urdu_hypothesis)
        english_hypothesis_path = Path(args.english_hypothesis)
        urdu_reference_path = Path(args.urdu_reference)
        english_reference_path = Path(args.english_reference)

        urdu_hypothesis_raw, urdu_duration = load_hypothesis_text(
            urdu_hypothesis_path, "urdu", max_seconds
        )
        english_hypothesis_raw, english_duration = load_hypothesis_text(
            english_hypothesis_path, "english", max_seconds
        )
        urdu_reference_raw = _read_text_file(urdu_reference_path)
        english_reference_raw = _read_text_file(english_reference_path)

        if max_seconds is not None and urdu_duration and english_duration:
            if urdu_duration.segment_count != english_duration.segment_count:
                raise EvaluationError(
                    "Segment count mismatch after duration filtering: "
                    f"Urdu {urdu_duration.segment_count} vs English {english_duration.segment_count}"
                )

        urdu_scoring_hyp, urdu_scoring_ref, urdu_alignment, urdu_hyp_norm, urdu_ref_norm = (
            prepare_comparison_pair(
                urdu_hypothesis_raw,
                urdu_reference_raw,
                "urdu",
                align=align,
                section_name=args.section,
                english_reference_raw=english_reference_raw,
            )
        )
        english_scoring_hyp, english_scoring_ref, english_alignment, english_hyp_norm, english_ref_norm = (
            prepare_comparison_pair(
                english_hypothesis_raw,
                english_reference_raw,
                "english",
                align=align,
                section_name=args.section,
                english_reference_raw=english_reference_raw,
            )
        )

        if not args.section:
            validate_section_alignment(urdu_ref_norm, english_ref_norm)

        urdu_metrics = compute_wer_cer(urdu_scoring_ref, urdu_scoring_hyp)
        english_metrics = compute_wer_cer(english_scoring_ref, english_scoring_hyp)

        if args.section:
            mode = f"section: {args.section}"
        elif max_seconds is not None:
            mode = "duration_window + anchor alignment" if align else "duration_window"
        elif align:
            mode = "full hypothesis + anchor alignment"
        else:
            mode = "full document"

        save_artifacts(
            output_dir,
            urdu_ref_norm,
            english_ref_norm,
            urdu_hyp_norm,
            english_hyp_norm,
            urdu_reference_aligned=urdu_scoring_ref if align else None,
            english_reference_aligned=english_scoring_ref if align else None,
        )

        results = build_results_payload(
            urdu_metrics,
            english_metrics,
            urdu_hypothesis_path,
            urdu_reference_path,
            english_hypothesis_path,
            english_reference_path,
            mode=mode,
            duration_info=urdu_duration,
            urdu_alignment=urdu_alignment,
            english_alignment=english_alignment,
            section_name=args.section,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        results_path.write_text(
            json.dumps(results, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        print_results_table(
            urdu_metrics,
            english_metrics,
            urdu_hypothesis_path,
            urdu_reference_path,
            english_hypothesis_path,
            english_reference_path,
            mode=mode,
            duration_info=urdu_duration,
            urdu_alignment=urdu_alignment,
            english_alignment=english_alignment,
            section_name=args.section,
        )
        print_normalization_log()
        print(f"Saved cleaned artifacts to: {output_dir}")
        print(f"Saved results to: {results_path}")

    except EvaluationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
