# ============================================================
# main.py — Run the complete Urdu Interview Pipeline
#
# Usage:
#   python main.py audio/your_interview.mp3
#   python main.py audio/your_interview.mp3 --start-stage 3
# ============================================================

import sys
import os
import argparse
import traceback

# Force UTF-8 output on Windows so Unicode symbols (✔ ⚠ etc.) print correctly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Ensure project root is on path ───────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from pipeline.utils import ensure_dirs, load_json, now_str

# ── Create all output directories ────────────────────────────
ensure_dirs(
    config.STAGE1_DIR, config.STAGE2_DIR, config.STAGE3_DIR,
    config.STAGE4_DIR, config.STAGE5_DIR, config.STAGE6_DIR,
)


def run_pipeline(audio_path: str, start_stage: int = 1):
    """
    Run all 6 pipeline stages in sequence.

    Args:
        audio_path   : Path to the input Urdu audio file
        start_stage  : Which stage to start from (1-6).
                       Stages 2-6 require the previous stage's JSON to exist.
    """
    print("\n" + "=" * 60)
    print("  URDU INTERVIEW PROCESSING PIPELINE")
    print("=" * 60)
    print(f"  Audio file  : {audio_path}")
    print(f"  Start stage : {start_stage}")
    print(f"  Started at  : {now_str()}")

    # ── Helper: derive interview_id from audio path ───────────
    from pipeline.utils import get_interview_id
    interview_id = get_interview_id(audio_path)

    # ── Stage result carrier ──────────────────────────────────
    stage1_result = None
    stage2_result = None
    stage3_result = None
    stage4_result = None
    stage5_result = None

    # If resuming from a later stage, load previous outputs
    if start_stage > 1:
        stage1_path = os.path.join(config.STAGE1_DIR, f"{interview_id}_urdu_transcript.json")
        if os.path.exists(stage1_path):
            stage1_result = load_json(stage1_path)
            print(f"\n  Loaded Stage 1 output from: {stage1_path}")

    if start_stage > 2:
        stage2_path = os.path.join(config.STAGE2_DIR, f"{interview_id}_verified_transcript.json")
        if os.path.exists(stage2_path):
            stage2_result = load_json(stage2_path)
            print(f"  Loaded Stage 2 output from: {stage2_path}")

    if start_stage > 3:
        stage3_path = os.path.join(config.STAGE3_DIR, f"{interview_id}_english_translation.json")
        if os.path.exists(stage3_path):
            stage3_result = load_json(stage3_path)
            print(f"  Loaded Stage 3 output from: {stage3_path}")

    if start_stage > 4:
        stage4_path = os.path.join(config.STAGE4_DIR, f"{interview_id}_verified_translation.json")
        if os.path.exists(stage4_path):
            stage4_result = load_json(stage4_path)
            print(f"  Loaded Stage 4 output from: {stage4_path}")

    if start_stage > 5:
        stage5_path = os.path.join(config.STAGE5_DIR, f"{interview_id}_deidentified.json")
        if os.path.exists(stage5_path):
            stage5_result = load_json(stage5_path)
            print(f"  Loaded Stage 5 output from: {stage5_path}")

    # ══════════════════════════════════════════════════════════
    # STAGE 1 — Urdu Transcription
    # ══════════════════════════════════════════════════════════
    if start_stage <= 1:
        try:
            from pipeline.transcribe import transcribe
            stage1_result = transcribe(audio_path)
        except Exception as e:
            print(f"\n❌ STAGE 1 FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # STAGE 2 — Verify Urdu Transcript
    # ══════════════════════════════════════════════════════════
    if start_stage <= 2:
        if stage1_result is None:
            print("\n❌ Stage 1 result not available. Run from stage 1.")
            sys.exit(1)
        try:
            from pipeline.verify_transcript import verify_transcript
            stage2_result = verify_transcript(stage1_result)
        except Exception as e:
            print(f"\n❌ STAGE 2 FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # STAGE 3 — Urdu → English Translation
    # ══════════════════════════════════════════════════════════
    if start_stage <= 3:
        if stage2_result is None:
            print("\n❌ Stage 2 result not available. Run from stage 2 or earlier.")
            sys.exit(1)
        try:
            from pipeline.translate import translate
            stage3_result = translate(stage2_result)
        except Exception as e:
            print(f"\n❌ STAGE 3 FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # STAGE 4 — Verify English Translation
    # ══════════════════════════════════════════════════════════
    if start_stage <= 4:
        if stage3_result is None:
            print("\n❌ Stage 3 result not available. Run from stage 3 or earlier.")
            sys.exit(1)
        try:
            from pipeline.verify_translation import verify_translation
            stage4_result = verify_translation(stage3_result)
        except Exception as e:
            print(f"\n❌ STAGE 4 FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # STAGE 5 — De-identification
    # ══════════════════════════════════════════════════════════
    if start_stage <= 5:
        if stage4_result is None:
            print("\n❌ Stage 4 result not available. Run from stage 4 or earlier.")
            sys.exit(1)
        try:
            from pipeline.deidentify import deidentify
            stage5_result = deidentify(stage4_result)
        except Exception as e:
            print(f"\n❌ STAGE 5 FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # STAGE 6 — Final Export (JSON + DOCX)
    # ══════════════════════════════════════════════════════════
    if start_stage <= 6:
        if stage5_result is None:
            print("\n❌ Stage 5 result not available. Run from stage 5 or earlier.")
            sys.exit(1)
        try:
            from pipeline.export import export
            final_result = export(stage5_result)
        except Exception as e:
            print(f"\n❌ STAGE 6 FAILED: {e}")
            traceback.print_exc()
            sys.exit(1)

    # ══════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE!")
    print(f"  Outputs saved in: {config.OUTPUT_DIR}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Urdu Interview Processing Pipeline"
    )
    parser.add_argument(
        "audio_path",
        type=str,
        help="Path to the input Urdu audio file (mp3/wav/m4a)"
    )
    parser.add_argument(
        "--start-stage",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="Stage to start from (default: 1). Useful to resume after failure."
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio_path):
        print(f"❌ Audio file not found: {args.audio_path}")
        sys.exit(1)

    run_pipeline(args.audio_path, start_stage=args.start_stage)
