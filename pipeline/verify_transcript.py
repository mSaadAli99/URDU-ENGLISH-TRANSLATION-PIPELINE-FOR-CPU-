# ============================================================
# pipeline/verify_transcript.py — Stage 2: Transcript Verification
# Checks confidence scores, flags low-quality segments,
# reports language breakdown (Urdu / English / mixed).
# ============================================================

import os
import sys
from pipeline.utils import save_json, print_banner, now_str

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def verify_transcript(stage1_result: dict) -> dict:
    """
    Stage 2: Verify transcript quality using Whisper confidence scores.

    Also reports the Urdu/English language split detected in Stage 1.
    """
    print_banner(2, "VERIFICATION OF TRANSCRIPT")

    segments     = stage1_result["segments"]
    interview_id = stage1_result["interview_id"]
    threshold    = config.CONFIDENCE_THRESHOLD

    print(f"  Interview ID        : {interview_id}")
    print(f"  Total segments      : {len(segments)}")
    print(f"  Urdu segments       : {stage1_result.get('urdu_segments', '?')}")
    print(f"  English segments    : {stage1_result.get('english_segments', '?')}")
    print(f"  Loops removed       : {stage1_result.get('loops_removed', 0)}")
    print(f"  Confidence threshold: {threshold}")

    verified_segments = []
    flagged_segments  = []
    confidence_scores = []

    for seg in segments:
        conf   = seg.get("confidence", 0.0)
        is_low = bool(conf < threshold)

        verified_seg = {
            **seg,
            "verified_text": f"[LOW CONFIDENCE] {seg['text']}".strip() if is_low else seg["text"],
            "flagged"      : bool(is_low),
            "flag_reason"  : f"Confidence {conf:.2f} below threshold {threshold}" if is_low else None,
        }

        verified_segments.append(verified_seg)
        confidence_scores.append(conf)

        if is_low:
            flagged_segments.append({
                "segment_id": seg["segment_id"],
                "start_fmt" : seg["start_fmt"],
                "end_fmt"   : seg["end_fmt"],
                "text"      : seg["text"],
                "confidence": conf,
                "language"  : seg.get("language", "unknown"),
                "reason"    : verified_seg["flag_reason"],
            })

        icon = "⚠ FLAGGED" if is_low else "✔"
        lang = "UR" if seg.get("is_urdu") else "EN"
        print(f"    [{icon}][{lang}] seg {seg['segment_id']:03d} | conf={conf:.2f} | {seg['text'][:50]}...")

    # ── Quality score ─────────────────────────────────────────
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    pct_good       = sum(1 for c in confidence_scores if c >= threshold) / len(confidence_scores) if confidence_scores else 0.0
    quality_score  = round((avg_confidence * 0.6 + pct_good * 0.4) * 100, 1)

    verified_full_text = " ".join(s["verified_text"] for s in verified_segments)

    verification_report = {
        "total_segments"      : len(segments),
        "flagged_segments"    : len(flagged_segments),
        "good_segments"       : len(segments) - len(flagged_segments),
        "avg_confidence"      : round(avg_confidence, 4),
        "quality_score"       : quality_score,
        "quality_label"       : _quality_label(quality_score),
        "duration_covered_min": stage1_result["duration_minutes"],
        "confidence_threshold": threshold,
        "urdu_segments"       : stage1_result.get("urdu_segments", 0),
        "english_segments"    : stage1_result.get("english_segments", 0),
        "loops_removed"       : stage1_result.get("loops_removed", 0),
        "flagged_details"     : flagged_segments,
    }

    print(f"\n  ── Verification Report ──────────────────────────")
    print(f"  Total segments   : {len(segments)}")
    print(f"  Flagged segments : {len(flagged_segments)}")
    print(f"  Avg confidence   : {avg_confidence:.4f}")
    print(f"  Quality score    : {quality_score}/100 ({_quality_label(quality_score)})")

    if quality_score < config.MIN_QUALITY_SCORE:
        print(f"\n  ⚠ Quality score {quality_score} is below minimum {config.MIN_QUALITY_SCORE}.")

    result = {
        "stage"                : 2,
        "stage_name"           : "Verification of Transcript",
        "interview_id"         : interview_id,
        "audio_filename"       : stage1_result["audio_filename"],
        "processed_at"         : now_str(),
        "original_full_text"   : stage1_result["full_urdu_text"],
        "verified_full_text"   : verified_full_text,
        "quality_score"        : quality_score,
        "quality_label"        : _quality_label(quality_score),
        "verification_report"  : verification_report,
        "verified_segments"    : verified_segments,
        "duration_minutes"     : stage1_result["duration_minutes"],
        "audio_path"           : stage1_result["audio_path"],
    }

    out_path = os.path.join(config.STAGE2_DIR, f"{interview_id}_verified_transcript.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 2 complete.")
    return result


def _quality_label(score: float) -> str:
    if score >= 85: return "Excellent"
    if score >= 70: return "Good"
    if score >= 55: return "Fair"
    if score >= 40: return "Poor"
    return "Very Poor"
