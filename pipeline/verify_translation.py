# ============================================================
# pipeline/verify_translation.py — Stage 4: Translation Verification
# Checks: empty, error markers, length ratio, nonsense detection,
#         non-ASCII output for English pass-through segments.
# ============================================================

import os
import re
import sys
from pipeline.utils import save_json, print_banner, now_str

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Common patterns that indicate NLLB / OPUS generated nonsense
_NONSENSE_RE = re.compile(
    r"\b(BAER:|BROLOD|OLOCTO|DECTRO|HOCKS|LAUNE|left-mani|Groups of drags)\b",
    re.IGNORECASE,
)

# Checks whether a string is mostly uppercase non-word tokens (junk output)
def _looks_like_junk(text: str) -> bool:
    if not text:
        return False
    words = text.split()
    if not words:
        return False
    caps = sum(1 for w in words if w.isupper() and len(w) >= 3)
    return caps / len(words) >= 0.5


def verify_translation(stage3_result: dict) -> dict:
    """
    Stage 4: Verify English translation quality.

    Pass-through segments (already English) are scored separately from
    translated Urdu segments so the overall score is meaningful.
    """
    print_banner(4, "VERIFICATION OF ENGLISH TRANSLATION")

    interview_id = stage3_result["interview_id"]
    segments     = stage3_result["translated_segments"]

    translated_count  = stage3_result.get("segments_translated", 0)
    passthrough_count = stage3_result.get("segments_passthrough", 0)

    print(f"  Interview ID      : {interview_id}")
    print(f"  Total segments    : {len(segments)}")
    print(f"  Translated        : {translated_count}")
    print(f"  Pass-through      : {passthrough_count}")
    print(f"\n  Checks: empty, error markers, length ratio, nonsense detection")

    verified_segments = []
    flagged_segments  = []
    quality_scores    = []

    for seg in segments:
        eng_text  = seg.get("english_text", "")
        urdu_text = seg.get("text", "")
        route     = seg.get("translation_route", "TRANSLATED")
        issues    = []
        is_flagged = False

        # Check 1: Empty
        if not eng_text or not eng_text.strip():
            issues.append("Empty translation")
            is_flagged = True

        # Check 2: Hard translation error marker
        elif "[TRANSLATION ERROR]" in eng_text:
            issues.append("Translation error occurred")
            is_flagged = True

        else:
            # Check 3: Nonsense / junk output detection (only for translated segments)
            if route == "TRANSLATED":
                if _NONSENSE_RE.search(eng_text):
                    issues.append("Nonsense output detected")
                    is_flagged = True
                elif _looks_like_junk(eng_text):
                    issues.append("Output looks like junk (mostly uppercase tokens)")
                    is_flagged = True

            # Check 4: Length ratio
            if urdu_text and eng_text and route == "TRANSLATED":
                ratio = len(eng_text) / max(len(urdu_text), 1)
                if ratio < 0.15:
                    issues.append(f"Translation suspiciously short (ratio={ratio:.2f})")
                    is_flagged = True
                elif ratio > 8.0:
                    issues.append(f"Translation suspiciously long (ratio={ratio:.2f})")
                    # Warn but don't flag — NLLB can be verbose

        # Score
        if is_flagged:
            seg_quality = 0
        elif issues:
            seg_quality = 70   # warnings only
        elif route == "PASS-THROUGH":
            seg_quality = 95   # pass-through is reliable by definition
        else:
            seg_quality = 100

        quality_scores.append(seg_quality)

        verified_text = f"[TRANSLATION FAILED] {eng_text}".strip() if is_flagged else eng_text

        verified_seg = {
            **seg,
            "verified_english"   : verified_text,
            "translation_flagged": bool(is_flagged),
            "translation_issues" : issues,
            "translation_quality": seg_quality,
        }
        verified_segments.append(verified_seg)

        if is_flagged:
            flagged_segments.append({
                "segment_id": seg["segment_id"],
                "start_fmt" : seg["start_fmt"],
                "urdu_text" : urdu_text,
                "eng_text"  : eng_text,
                "route"     : route,
                "issues"    : issues,
            })

        icon      = "⚠ FLAGGED" if is_flagged else "✔"
        route_tag = "[T]" if route == "TRANSLATED" else "[-]"
        issue_str = " | " + "; ".join(issues) if issues else ""
        print(f"    [{icon}]{route_tag} seg {seg['segment_id']:03d}{issue_str}")
        print(f"             EN: {eng_text[:70]}...")

    overall_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0.0

    verified_english_full = " ".join(s["verified_english"] for s in verified_segments)

    translation_report = {
        "total_segments"       : len(segments),
        "flagged_segments"     : len(flagged_segments),
        "good_segments"        : len(segments) - len(flagged_segments),
        "overall_quality_score": overall_quality,
        "quality_label"        : _quality_label(overall_quality),
        "segments_translated"  : translated_count,
        "segments_passthrough" : passthrough_count,
        "flagged_details"      : flagged_segments,
    }

    print(f"\n  ── Translation Verification Report ──────────────")
    print(f"  Total segments   : {len(segments)}")
    print(f"  Flagged          : {len(flagged_segments)}")
    print(f"  Quality score    : {overall_quality}/100 ({_quality_label(overall_quality)})")

    if overall_quality < config.MIN_TRANSLATION_SCORE:
        print(f"\n  ⚠ Quality {overall_quality} below minimum {config.MIN_TRANSLATION_SCORE}.")

    result = {
        "stage"                          : 4,
        "stage_name"                     : "Verification of English Translation",
        "interview_id"                   : interview_id,
        "audio_filename"                 : stage3_result["audio_filename"],
        "processed_at"                   : now_str(),
        "urdu_full_text"                 : stage3_result["urdu_full_text"],
        "english_full_text"              : stage3_result["english_full_text"],
        "verified_english_full"          : verified_english_full,
        "translation_quality_score"      : overall_quality,
        "translation_quality_label"      : _quality_label(overall_quality),
        "transcript_quality_score"       : stage3_result["transcript_quality"],
        "duration_minutes"               : stage3_result["duration_minutes"],
        "audio_path"                     : stage3_result["audio_path"],
        "translation_verification_report": translation_report,
        "transcript_verification_report" : stage3_result["transcript_verification_report"],
        "verified_segments"              : verified_segments,
    }

    out_path = os.path.join(config.STAGE4_DIR, f"{interview_id}_verified_translation.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 4 complete.")
    return result


def _quality_label(score: float) -> str:
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 55: return "Fair"
    if score >= 40: return "Poor"
    return "Very Poor"
