# ============================================================
# pipeline/transcribe.py — Stage 1: ASR (multilingual)
# Uses faster-whisper with whisper-large-v3-turbo.
# Supports pure Urdu, pure English, and code-switched audio.
# Language is AUTO-DETECTED per segment unless config forces one.
# Whisper hallucination/repetition loops are collapsed here.
# ============================================================

import os
import sys
from pipeline.utils import (
    save_json, get_interview_id, format_timestamp,
    print_banner, now_str, collapse_repetitions, is_urdu_text,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def transcribe(audio_path: str) -> dict:
    """
    Stage 1: Transcribe audio using faster-whisper.

    Language is auto-detected per segment when config.WHISPER_LANGUAGE is None,
    which is the correct setting for code-switched Urdu/English interviews.
    Whisper repetition loops are filtered before output.
    """
    print_banner(1, "TRANSCRIPTION (ASR)")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    lang_display = config.WHISPER_LANGUAGE or "auto-detect"
    print(f"  Audio file : {audio_path}")
    print(f"  Model      : whisper-{config.WHISPER_MODEL}")
    print(f"  Language   : {lang_display}")
    print(f"  Device     : {config.WHISPER_DEVICE}")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

    compute_type = config.WHISPER_COMPUTE_TYPE
    if config.WHISPER_DEVICE == "cpu" and compute_type == "float16":
        compute_type = "int8"
        print("  ⚠  float16 not supported on CPU — falling back to int8")

    print("\n  Loading Whisper model (first run downloads ~800MB)...")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=compute_type,
    )
    print("  ✔ Model loaded.")

    # ── Transcription params ──────────────────────────────────
    transcribe_kwargs = dict(
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        beam_size=config.WHISPER_BEAM_SIZE,
        temperature=config.WHISPER_TEMPERATURE,
        # condition_on_previous_text=False reduces hallucination loops
        condition_on_previous_text=False,
        initial_prompt=config.WHISPER_INITIAL_PROMPT,
    )

    # Only set language when explicitly configured — None triggers auto-detect
    if config.WHISPER_LANGUAGE is not None:
        transcribe_kwargs["language"] = config.WHISPER_LANGUAGE

    print(f"\n  Transcribing audio (language: {lang_display})...")
    segments_iter, info = model.transcribe(audio_path, **transcribe_kwargs)

    # ── Collect raw segments ──────────────────────────────────
    raw_segments = []
    for seg in segments_iter:
        if seg.words:
            confidences = [w.probability for w in seg.words if hasattr(w, "probability")]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        else:
            avg_conf = float(getattr(seg, "avg_logprob", -1.0))
            avg_conf = max(0.0, min(1.0, (avg_conf + 1.0)))  # rough -1..0 → 0..1

        # Tag segment language using Whisper's detected language (if available)
        seg_lang = getattr(seg, "language", None) or config.WHISPER_LANGUAGE or "unknown"

        raw_segments.append({
            "segment_id"  : seg.id,
            "start"       : round(seg.start, 3),
            "end"         : round(seg.end, 3),
            "start_fmt"   : format_timestamp(seg.start),
            "end_fmt"     : format_timestamp(seg.end),
            "text"        : seg.text.strip(),
            "confidence"  : round(avg_conf, 4),
            "language"    : seg_lang,
            "is_urdu"     : is_urdu_text(seg.text.strip()),
            "words"       : [
                {
                    "word"      : w.word,
                    "start"     : round(w.start, 3),
                    "end"       : round(w.end, 3),
                    "confidence": round(w.probability, 4) if hasattr(w, "probability") else None,
                }
                for w in (seg.words or [])
            ],
        })

    # ── Collapse Whisper hallucination loops ──────────────────
    before_filter = len(raw_segments)
    segments = collapse_repetitions(raw_segments, max_consecutive=config.REPETITION_MAX_CONSECUTIVE)
    removed_loops = before_filter - len(segments)
    if removed_loops:
        print(f"\n  ⚠  Collapsed {removed_loops} repeated hallucination segments.")

    # Re-number segment IDs after filtering
    for i, seg in enumerate(segments, start=1):
        seg["segment_id"] = i

    # ── Language breakdown ────────────────────────────────────
    urdu_count    = sum(1 for s in segments if s["is_urdu"])
    english_count = len(segments) - urdu_count

    full_text = " ".join(s["text"] for s in segments)
    duration_seconds = info.duration if hasattr(info, "duration") else (segments[-1]["end"] if segments else 0)

    print(f"\n  Processing complete:")
    print(f"  Segments kept    : {len(segments)} (dropped {removed_loops} loops)")
    print(f"  Urdu segments    : {urdu_count}")
    print(f"  English segments : {english_count}")
    for seg in segments:
        icon = "✔" if seg["confidence"] >= config.CONFIDENCE_THRESHOLD else "⚠"
        lang = "UR" if seg["is_urdu"] else "EN"
        print(f"    [{icon}][{lang}] {seg['start_fmt']} → {seg['end_fmt']} "
              f"| conf={seg['confidence']:.2f} | {seg['text'][:55]}...")

    result = {
        "stage"              : 1,
        "stage_name"         : "Transcription (ASR)",
        "interview_id"       : get_interview_id(audio_path),
        "audio_filename"     : os.path.basename(audio_path),
        "audio_path"         : audio_path,
        "processed_at"       : now_str(),
        "model"              : f"whisper-{config.WHISPER_MODEL}",
        "language_mode"      : lang_display,
        "duration_seconds"   : round(duration_seconds, 2),
        "duration_minutes"   : round(duration_seconds / 60, 2),
        "total_segments"     : len(segments),
        "urdu_segments"      : urdu_count,
        "english_segments"   : english_count,
        "loops_removed"      : removed_loops,
        "full_urdu_text"     : full_text,
        "segments"           : segments,
    }

    interview_id = result["interview_id"]
    out_path = os.path.join(config.STAGE1_DIR, f"{interview_id}_urdu_transcript.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 1 complete.")
    print(f"  Total segments : {len(segments)}  |  Duration: {result['duration_minutes']} min")
    return result
