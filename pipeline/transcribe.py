# ============================================================
# pipeline/transcribe.py — Stage 1: Urdu ASR
# Uses faster-whisper with whisper-large-v3-turbo
# Outputs: Urdu transcript with timestamps & confidence scores
# ============================================================

import os
import sys
from pipeline.utils import save_json, get_interview_id, format_timestamp, print_banner, now_str

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def transcribe(audio_path: str) -> dict:
    """
    Stage 1: Transcribe Urdu audio to text using faster-whisper.

    Args:
        audio_path: Path to the input audio file (.mp3 / .wav / .m4a)

    Returns:
        dict with full transcript data including segments and confidence scores
    """
    print_banner(1, "URDU TRANSCRIPTION (ASR)")

    # ── Validate input ────────────────────────────────────────
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"  Audio file : {audio_path}")
    print(f"  Model      : whisper-{config.WHISPER_MODEL}")
    print(f"  Language   : {config.WHISPER_LANGUAGE} (Urdu)")
    print(f"  Device     : {config.WHISPER_DEVICE}")

    # ── Load faster-whisper model ─────────────────────────────
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")

    print("\n  Loading Whisper model (first run downloads ~800MB)...")
    model = WhisperModel(
        config.WHISPER_MODEL,
        device=config.WHISPER_DEVICE,
        compute_type=config.WHISPER_COMPUTE_TYPE,
    )
    print("  ✔ Model loaded.")

    # ── Run transcription ─────────────────────────────────────
    print("\n  Transcribing audio... (this may take a few minutes)")
    segments_iter, info = model.transcribe(
        audio_path,
        language=config.WHISPER_LANGUAGE,  # Force Urdu
        word_timestamps=True,              # Word-level timestamps
        vad_filter=True,                   # Voice Activity Detection (removes silence)
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    # ── Collect segments ──────────────────────────────────────
    segments = []
    full_text_parts = []

    print("\n  Processing segments:")
    for seg in segments_iter:
        avg_word_confidence = 0.0

        # Calculate average confidence from word-level scores
        if seg.words:
            confidences = [w.probability for w in seg.words if hasattr(w, "probability")]
            avg_word_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        segment_data = {
            "segment_id"  : seg.id,
            "start"       : round(seg.start, 3),
            "end"         : round(seg.end, 3),
            "start_fmt"   : format_timestamp(seg.start),
            "end_fmt"     : format_timestamp(seg.end),
            "text"        : seg.text.strip(),
            "confidence"  : round(avg_word_confidence, 4),
            "words"       : [
                {
                    "word"      : w.word,
                    "start"     : round(w.start, 3),
                    "end"       : round(w.end, 3),
                    "confidence": round(w.probability, 4) if hasattr(w, "probability") else None,
                }
                for w in (seg.words or [])
            ],
        }

        segments.append(segment_data)
        full_text_parts.append(seg.text.strip())

        # Live progress print
        conf_icon = "✔" if avg_word_confidence >= config.CONFIDENCE_THRESHOLD else "⚠"
        print(f"    [{conf_icon}] {segment_data['start_fmt']} → {segment_data['end_fmt']} "
              f"| conf={avg_word_confidence:.2f} | {seg.text.strip()[:60]}...")

    full_urdu_text = " ".join(full_text_parts)
    duration_seconds = info.duration if hasattr(info, "duration") else (segments[-1]["end"] if segments else 0)

    # ── Build output dict ─────────────────────────────────────
    result = {
        "stage"             : 1,
        "stage_name"        : "Urdu Transcription",
        "interview_id"      : get_interview_id(audio_path),
        "audio_filename"    : os.path.basename(audio_path),
        "audio_path"        : audio_path,
        "processed_at"      : now_str(),
        "model"             : f"whisper-{config.WHISPER_MODEL}",
        "language"          : config.WHISPER_LANGUAGE,
        "duration_seconds"  : round(duration_seconds, 2),
        "duration_minutes"  : round(duration_seconds / 60, 2),
        "total_segments"    : len(segments),
        "full_urdu_text"    : full_urdu_text,
        "segments"          : segments,
    }

    # ── Save output ───────────────────────────────────────────
    interview_id = result["interview_id"]
    out_path = os.path.join(config.STAGE1_DIR, f"{interview_id}_urdu_transcript.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 1 complete.")
    print(f"  Total segments : {len(segments)}")
    print(f"  Duration       : {result['duration_minutes']} min")

    return result
