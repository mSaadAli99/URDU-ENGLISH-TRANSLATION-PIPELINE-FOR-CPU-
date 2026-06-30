# ============================================================
# pipeline/translate.py — Stage 3: Urdu → English Translation
# Uses facebook/nllb-200-distilled-600M
# Handles long text via sentence-level chunking
# ============================================================

import os
from pipeline.utils import save_json, chunk_text, split_sentences, chunk_sentences, fix_translation_errors, print_banner, now_str

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def translate(stage2_result: dict) -> dict:
    """
    Stage 3: Translate verified Urdu transcript to English using NLLB-200.

    Args:
        stage2_result: Output dict from Stage 2 (verify_transcript.py)

    Returns:
        dict with English translations alongside Urdu originals per segment
    """
    print_banner(3, "TRANSLATION: URDU → ENGLISH")

    interview_id = stage2_result["interview_id"]

    print(f"  Interview ID   : {interview_id}")
    print(f"  Model          : {config.TRANSLATION_MODEL}")
    print(f"  Source lang    : {config.NLLB_SRC_LANG} (Urdu)")
    print(f"  Target lang    : {config.NLLB_TGT_LANG} (English)")
    print(f"  Chunk size     : {config.CHUNK_SIZE} chars")

    # ── Load NLLB-200 model ───────────────────────────────────
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except ImportError:
        raise ImportError("transformers / torch not installed. Run: pip install transformers torch")

    print("\n  Loading NLLB-200 model (first run downloads ~1.2GB)...")
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSLATION_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(config.TRANSLATION_MODEL)

    # Move model to GPU if available
    device = "cuda" if config.WHISPER_DEVICE == "cuda" else "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            model = model.to("cuda")
            device = "cuda"
    except Exception:
        device = "cpu"

    model.eval()
    print(f"  ✔ Model loaded on {device}.")

    # ── Translation helper ────────────────────────────────────
    def translate_text(urdu_text: str) -> str:
        """Translate a single Urdu text string to English using NLLB."""
        if not urdu_text.strip():
            return ""

        # Split into sentences first, then chunk to preserve context
        sentences = split_sentences(urdu_text)
        chunks = chunk_sentences(sentences, max_chars=config.CHUNK_SIZE)
        translated_chunks = []

        for chunk in chunks:
            try:
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)

                # Get target language token id
                tgt_lang_id = tokenizer.convert_tokens_to_ids(config.NLLB_TGT_LANG)

                with __import__("torch").no_grad():
                    outputs = model.generate(
                        **inputs,
                        forced_bos_token_id=tgt_lang_id,
                        max_new_tokens=512,
                        num_beams=5,                  # Increased from 4 for better quality
                        early_stopping=True,
                        temperature=0.7,              # Add temperature for better diversity
                        top_p=0.95,                   # Nucleus sampling for coherence
                    )

                translated = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                # Fix common errors in translation
                translated = fix_translation_errors(translated.strip())
                translated_chunks.append(translated)

            except Exception as e:
                print(f"    ⚠ Chunk translation failed: {e}")
                translated_chunks.append("[TRANSLATION ERROR]")

        return " ".join(translated_chunks)

    # ── Translate full text ───────────────────────────────────
    print("\n  Translating full text...")
    verified_full_text = stage2_result.get("verified_full_text", "")
    full_english_text  = translate_text(verified_full_text)

    # ── Translate segment by segment with context ─────────────
    print("\n  Translating segments:")
    translated_segments = []
    
    for idx, seg in enumerate(stage2_result["verified_segments"]):
        urdu_text = seg["text"]
        
        # Build context from previous segments (if enabled)
        context_prefix = ""
        if config.USE_CONTEXT_WINDOW and idx > 0:
            # Look back N segments for context
            start_idx = max(0, idx - config.CONTEXT_WINDOW_SIZE)
            context_segs = stage2_result["verified_segments"][start_idx:idx]
            context_texts = [s["text"] for s in context_segs]
            context_prefix = " ".join(context_texts) + " "
        
        # Translate with context
        context_text = context_prefix + urdu_text
        eng_text = translate_text(context_text)
        
        # Remove context from output if it was added
        if config.USE_CONTEXT_WINDOW and context_prefix:
            # The translation should be mostly the last part, but we only keep new translation
            # This is heuristic - typically output correlates to input length
            parts = eng_text.split()
            # Estimate how many words were context
            context_word_count = len(context_prefix.split())
            output_start = max(0, len(parts) - max(1, len(urdu_text.split())))
            eng_text = " ".join(parts[output_start:]).strip() if output_start < len(parts) else eng_text

        translated_seg = {
            **seg,
            "english_text": eng_text,
        }
        translated_segments.append(translated_seg)

        # Progress with memory info
        print(f"    ✔ seg {seg['segment_id']:03d}/{len(stage2_result['verified_segments'])} | UR: {urdu_text[:40]}...")
        print(f"             | EN: {eng_text[:60]}...")

    # ── Build result dict ─────────────────────────────────────
    result = {
        "stage"               : 3,
        "stage_name"          : "Translation: Urdu → English",
        "interview_id"        : interview_id,
        "audio_filename"      : stage2_result["audio_filename"],
        "processed_at"        : now_str(),
        "translation_model"   : config.TRANSLATION_MODEL,
        "urdu_full_text"      : verified_full_text,
        "english_full_text"   : full_english_text,
        "transcript_quality"  : stage2_result["quality_score"],
        "duration_minutes"    : stage2_result["duration_minutes"],
        "audio_path"          : stage2_result["audio_path"],
        "translated_segments" : translated_segments,
        # Pass verification report forward
        "transcript_verification_report": stage2_result["verification_report"],
    }

    # ── Save output ───────────────────────────────────────────
    out_path = os.path.join(config.STAGE3_DIR, f"{interview_id}_english_translation.json")
    save_json(result, out_path)

    print(f"\n  ✔ Stage 3 complete.")
    print(f"  Segments translated: {len(translated_segments)}")
    return result
