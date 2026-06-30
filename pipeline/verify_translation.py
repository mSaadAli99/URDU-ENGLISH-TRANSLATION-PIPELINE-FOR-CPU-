# ============================================================
# pipeline/verify_translation.py — Stage 4: Translation Verification
# Uses langdetect to confirm English output
# Flags failed/poor translations
# Generates quality report
# ============================================================

import os
from pipeline.utils import save_json, print_banner, now_str

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def verify_translation(stage3_result: dict) -> dict:
    """
    Stage 4: Verify quality of English translations.

    Args:
        stage3_result: Output dict from Stage 3 (translate.py)

    Returns:
        dict with verified translations, flags, and quality report
    """
    print_banner(4, "VERIFICATION OF ENGLISH TRANSLATION")

    interview_id = stage3_result["interview_id"]
    segments     = stage3_result["translated_segments"]

    print(f"  Interview ID   : {interview_id}")
    print(f"  Total segments : {len(segments)}")

    # ── Load langdetect ───────────────────────────────────────
    try:
        from langdetect import detect, LangDetectException
        langdetect_available = True
    except ImportError:
        print("  ⚠ langdetect not installed. Language detection skipped.")
        langdetect_available = False

    # ── Verify each segment ───────────────────────────────────
    verified_segments = []
    flagged_segments  = []
    quality_scores    = []

    for seg in segments:
        eng_text    = seg.get("english_text", "")
        issues      = []
        is_flagged  = False

        # Check 1: Empty translation
        if not eng_text or eng_text.strip() == "":
            issues.append("Empty translation")
            is_flagged = True

        # Check 2: Translation error marker
        elif "[TRANSLATION ERROR]" in eng_text:
            issues.append("Translation error occurred")
            is_flagged = True

        # Check 3: Language detection — confirm output is English
        elif langdetect_available and len(eng_text.split()) >= 3:
            try:
                detected_lang = detect(eng_text)
                if detected_lang != "en":
                    issues.append(f"Detected language is '{detected_lang}', expected 'en'")
                    is_flagged = True
            except LangDetectException:
                issues.append("Language detection failed (text too short or ambiguous)")

        # Check 4: Very short translation vs Urdu original
        urdu_text = seg.get("text", "")
        if urdu_text and eng_text:
            ratio = len(eng_text) / max(len(urdu_text), 1)
            # Flag if English is <20% or >500% of Urdu length (likely garbage)
            if ratio < 0.2:
                issues.append(f"Translation suspiciously short (ratio={ratio:.2f})")
                is_flagged = True
            elif ratio > 5.0:
                issues.append(f"Translation suspiciously long (ratio={ratio:.2f})")

        # Assign segment quality score (0-100)
        seg_quality = 0 if is_flagged else 100
        if issues and not is_flagged:
            seg_quality = 70  # Minor warnings, not flagged

        quality_scores.append(seg_quality)

        # Build verified segment
        verified_text = f"[TRANSLATION FAILED] {eng_text}".strip() if is_flagged else eng_text

        verified_seg = {
            **seg,
            "verified_english": verified_text,
            "translation_flagged": bool(is_flagged),  # Ensure Python bool
            "translation_issues": issues,
            "translation_quality": seg_quality,
        }
        verified_segments.append(verified_seg)

        if is_flagged:
            flagged_segments.append({
                "segment_id": seg["segment_id"],
                "start_fmt" : seg["start_fmt"],
                "urdu_text" : seg.get("text", ""),
                "eng_text"  : eng_text,
                "issues"    : issues,
            })

        icon = "⚠ FLAGGED" if is_flagged else "✔"
        issue_str = " | " + "; ".join(issues) if issues else ""
        print(f"    [{icon}] seg {seg['segment_id']:03d}{issue_str}")
        print(f"             EN: {eng_text[:70]}...")

    # ── Overall quality score ─────────────────────────────────
    overall_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0.0

    # ── Build verified full English text ──────────────────────
    verified_english_full = " ".join(
        s["verified_english"] for s in verified_segments
    )

    # ── Translation verification report ──────────────────────
    translation_report = {
        "total_segments"        : len(segments),
        "flagged_segments"      : len(flagged_segments),
        "good_segments"         : len(segments) - len(flagged_segments),
        "overall_quality_score" : overall_quality,
        "quality_label"         : _quality_label(overall_quality),
        "flagged_details"       : flagged_segments,
    }

    # ── Print report ──────────────────────────────────────────
    print(f"\n  ── Translation Verification Report ──────────")
    print(f"  Total segments    : {len(segments)}")
    print(f"  Flagged           : {len(flagged_segments)}")
    print(f"  Quality score     : {overall_quality}/100 ({_quality_label(overall_quality)})")

    if overall_quality < config.MIN_TRANSLATION_SCORE:
        print(f"\n  ⚠ WARNING: Translation quality {overall_quality} is below minimum {config.MIN_TRANSLATION_SCORE}.")
        print(f"  Consider reviewing flagged segments manually.")

    # ── Build result dict ─────────────────────────────────────
    result = {
        "stage"                        : 4,
        "stage_name"                   : "Verification of English Translation",
        "interview_id"                 : interview_id,
        "audio_filename"               : stage3_result["audio_filename"],
        "processed_at"                 : now_str(),
        "urdu_full_text"               : stage3_result["urdu_full_text"],
        "english_full_text"            : stage3_result["english_full_text"],
        "verified_english_full"        : verified_english_full,
        "translation_quality_score"    : overall_quality,
        "translation_quality_label"    : _quality_label(overall_quality),
        "transcript_quality_score"     : stage3_result["transcript_quality"],
        "duration_minutes"             : stage3_result["duration_minutes"],
        "audio_path"                   : stage3_result["audio_path"],
        "translation_verification_report": translation_report,
        "transcript_verification_report" : stage3_result["transcript_verification_report"],
        "verified_segments"            : verified_segments,
    }

    # ── Save output ───────────────────────────────────────────
    out_path = os.path.join(config.STAGE4_DIR, f"{interview_id}_verified_translation.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 4 complete.")
    return result


def _quality_label(score: float) -> str:
    if score >= 85:  return "Excellent"
    if score >= 70:  return "Good"
    if score >= 55:  return "Fair"
    if score >= 40:  return "Poor"
    return "Very Poor"
