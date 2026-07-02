# ============================================================
# config.py — Central configuration for Urdu Pipeline
# All model names, paths, thresholds in one place
# ============================================================

import os

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

AUDIO_DIR       = os.path.join(BASE_DIR, "audio")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")

STAGE1_DIR = os.path.join(OUTPUT_DIR, "1_urdu_transcripts")
STAGE2_DIR = os.path.join(OUTPUT_DIR, "2_verified_transcripts")
STAGE3_DIR = os.path.join(OUTPUT_DIR, "3_english_translations")
STAGE4_DIR = os.path.join(OUTPUT_DIR, "4_verified_translations")
STAGE5_DIR = os.path.join(OUTPUT_DIR, "5_deidentified")
STAGE6_DIR = os.path.join(OUTPUT_DIR, "6_final_dataset")

# ── Stage 1: ASR Model ───────────────────────────────────────
WHISPER_MODEL        = "large-v3-turbo"   # openai/whisper-large-v3-turbo via faster-whisper
# None = auto-detect language per segment (correct for code-switched Urdu/English interviews)
# Set to "ur" only if the entire audio is Urdu with no English
WHISPER_LANGUAGE     = None
# Neutral prompt for mixed interviews — does NOT bias toward one language
WHISPER_INITIAL_PROMPT = "This is a research interview. The speakers may use both Urdu and English."
WHISPER_TEMPERATURE  = 0.0                 # 0 = greedy / most deterministic; reduces hallucination loops
WHISPER_BEAM_SIZE    = 5                   # Higher beam = better quality

# ── Stage 1: Repetition / hallucination filter ────────────────
# Whisper sometimes loops the same phrase; collapse runs longer than this
REPETITION_MAX_CONSECUTIVE = 3   # Flag a segment if same text appeared N+ times in a row

# Auto-detect GPU; fall back to CPU gracefully
try:
    import torch as _torch
    if _torch.cuda.is_available():
        WHISPER_DEVICE       = "cuda"
        WHISPER_COMPUTE_TYPE = "float16"   # float16 is fast on GPU
    else:
        WHISPER_DEVICE       = "cpu"
        WHISPER_COMPUTE_TYPE = "int8"      # int8 is the only quantisation faster-whisper supports on CPU
except ImportError:
    WHISPER_DEVICE       = "cpu"
    WHISPER_COMPUTE_TYPE = "int8"

# ── Stage 2: Transcript Verification ─────────────────────────
# Lowered from 0.75 — mixed-language audio legitimately scores 0.60-0.75
CONFIDENCE_THRESHOLD = 0.60    # Flag segments below this confidence
MIN_QUALITY_SCORE    = 60      # Minimum acceptable transcript quality (0-100)

# ── Stage 3: Translation Model ───────────────────────────────
TRANSLATION_MODEL    = "facebook/nllb-200-1.3B"  # Best quality — use on Colab GPU
# Lighter fallback for local CPU when RAM is limited (~300 MB vs ~2.5 GB)
CPU_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-ur-en"
NLLB_SRC_LANG        = "urd_Arab"   # Urdu source language code for NLLB
NLLB_TGT_LANG        = "eng_Latn"   # English target language code for NLLB
CHUNK_SIZE           = 500           # Max characters per sentence chunk (for sentence-based splitting)
BATCH_TRANSLATION_SIZE = 5            # Translate N segments in one batch (speeds up long audio 5-10x)
USE_CONTEXT_WINDOW   = True           # Use previous segment for context (better continuity)
CONTEXT_WINDOW_SIZE  = 2              # Look back N segments for context

# ── Stage 4: Translation Verification ────────────────────────
MIN_TRANSLATION_SCORE = 50     # Minimum acceptable translation quality (0-100)

# ── Stage 5: De-identification ───────────────────────────────
SPACY_MODEL = "en_core_web_lg"

# Entity type → replacement tag mapping
ENTITY_REPLACEMENTS = {
    "PERSON":       "[NAME]",
    "PHONE_NUMBER": "[PHONE]",
    "EMAIL_ADDRESS":"[EMAIL]",
    "LOCATION":     "[LOCATION]",
    "GPE":          "[LOCATION]",
    "ORG":          "[ORGANIZATION]",
    "NRP":          "[ORGANIZATION]",
    "DATE_TIME":    "[DATE]",
    "US_SSN":       "[ID]",
    "US_PASSPORT":  "[ID]",
    "IBAN_CODE":    "[ID]",
    "CREDIT_CARD":  "[ID]",
    "URL":          "[URL]",
    "IP_ADDRESS":   "[ID]",
}

# ── Stage 6: Export ───────────────────────────────────────────
DOCX_TITLE         = "Urdu Interview — De-identified Qualitative Dataset"
DOCX_AUTHOR        = "Urdu Pipeline"
FINAL_JSON_INDENT  = 2
