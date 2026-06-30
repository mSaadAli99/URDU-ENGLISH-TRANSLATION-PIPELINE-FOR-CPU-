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
WHISPER_LANGUAGE     = "ur"               # Force Urdu transcription
WHISPER_DEVICE       = "cuda"             # "cuda" on Colab GPU, "cpu" locally
WHISPER_COMPUTE_TYPE = "float16"          # float16 on GPU, int8 on CPU

# ── Stage 2: Transcript Verification ─────────────────────────
CONFIDENCE_THRESHOLD = 0.75    # Flag segments below this confidence
MIN_QUALITY_SCORE    = 60      # Minimum acceptable transcript quality (0-100)

# ── Stage 3: Translation Model ───────────────────────────────
TRANSLATION_MODEL    = "facebook/nllb-200-1.3B"  # Upgraded from 600M for better quality
NLLB_SRC_LANG        = "urd_Arab"   # Urdu source language code for NLLB
NLLB_TGT_LANG        = "eng_Latn"   # English target language code for NLLB
CHUNK_SIZE           = 500           # Max characters per sentence chunk (for sentence-based splitting)

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
