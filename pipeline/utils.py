# ============================================================
# pipeline/utils.py — Shared helper functions
# ============================================================

import os
import json
import re
from datetime import datetime
import numpy as np


def ensure_dirs(*dirs):
    """Create output directories if they don't exist."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_json(data: dict, path: str, indent: int = 2):
    """Save a dictionary to a JSON file with numpy type support."""
    ensure_dirs(os.path.dirname(path))
    # Convert all numpy types before saving
    data_converted = convert_numpy_types(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_converted, f, ensure_ascii=False, indent=indent, cls=NumpyEncoder)
    print(f"  ✔ Saved → {path}")


def load_json(path: str) -> dict:
    """Load a JSON file and return as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_interview_id(audio_path: str) -> str:
    """Generate a clean interview ID from the audio filename."""
    basename = os.path.basename(audio_path)
    name, _ = os.path.splitext(basename)
    # Replace spaces and special chars with underscore
    clean = re.sub(r"[^\w]", "_", name)
    return clean


def chunk_text(text: str, max_chars: int = 400) -> list:
    """
    Split long text into chunks of max_chars.
    Tries to split on sentence boundaries (. ! ?) to keep meaning intact.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?۔؟])\s+', text.strip())

    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # If single sentence > max_chars, hard-split it
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS.mmm format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def print_banner(stage_num: int, stage_name: str):
    """Print a clear stage banner to terminal."""
    print("\n" + "=" * 60)
    print(f"  STAGE {stage_num}: {stage_name}")
    print("=" * 60)


def now_str() -> str:
    """Return current datetime as ISO string."""
    return datetime.now().isoformat()


def split_sentences(text: str) -> list:
    """
    Split text into sentences, respecting Urdu and English punctuation.
    Handles: . ! ? ۔ ؟ and similar markers.
    """
    # Replace Urdu punctuation with English equivalents for consistency
    text = text.replace('۔', '.').replace('؟', '?')
    
    # Split on sentence boundaries: space after punctuation
    # Lookahead regex: split on . ! ? followed by space
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Clean up empty sentences
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences: list, max_chars: int = 500) -> list:
    """
    Combine sentences into chunks without exceeding max_chars.
    Preserves sentence boundaries for better translation context.
    
    Args:
        sentences: List of sentence strings
        max_chars: Maximum characters per chunk
    
    Returns:
        List of chunked text strings
    """
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Add sentence to chunk if it fits
        test_chunk = current_chunk + (" " if current_chunk else "") + sentence
        
        if len(test_chunk) <= max_chars:
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If single sentence exceeds max_chars, it must be chunked
            if len(sentence) > max_chars:
                # Hard split at character boundary
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                current_chunk = ""
            else:
                current_chunk = sentence
    
    # Append remaining chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
