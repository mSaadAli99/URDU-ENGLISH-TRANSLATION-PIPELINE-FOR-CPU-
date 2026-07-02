#!/usr/bin/env python3
"""Quality report for pipeline outputs."""
import json, sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

d  = json.load(open("outputs/6_final_dataset/test_audio_final_dataset.json", encoding="utf-8"))
s1 = json.load(open("outputs/1_urdu_transcripts/test_audio_urdu_transcript.json", encoding="utf-8"))
s3 = json.load(open("outputs/3_english_translations/test_audio_english_translation.json", encoding="utf-8"))

print("=" * 55)
print("  PIPELINE QUALITY REPORT")
print("=" * 55)
print(f"  Audio              : {d['audio_filename']}")
print(f"  Duration           : {d['duration_minutes']} min")
print(f"  Language mode      : {s1.get('language_mode', 'N/A')}")
print(f"  Hallucination loops removed: {s1.get('loops_removed', 0)}")
print()
print(f"  Segments total     : {s1.get('total_segments')}")
print(f"  Urdu segments      : {s1.get('urdu_segments')}")
print(f"  English segments   : {s1.get('english_segments')}")
print()
print(f"  Transcript quality : {d['transcript_quality_score']}/100  "
      f"({d.get('transcript_verification_report',{}).get('quality_label','')})")
print(f"  Translation quality: {d['translation_quality_score']}/100  "
      f"({d.get('translation_verification_report',{}).get('quality_label','')})")
print()
print(f"  Segments translated (Urdu only): {s3.get('segments_translated')}")
print(f"  Segments pass-through (English): {s3.get('segments_passthrough')}")
print()
print(f"  Entities removed   : {d['entities_removed_count']}")
print(f"  By type            : {d['entities_removed_by_type']}")
print()

tr = d.get("transcript_verification_report", {})
tv = d.get("translation_verification_report", {})
print(f"  Transcript flagged : {tr.get('flagged_segments')}/{tr.get('total_segments')}")
print(f"  Translation flagged: {tv.get('flagged_segments')}/{tv.get('total_segments')}")
print()

# Hallucination checks on final text
text = d.get("english_translation", "")
loops = [
    "the course of the course",
    "that is something that is something",
    "I will do this. I will do this",
]
found = [l for l in loops if l in text]
print("  Hallucination loops in output:", found if found else "None  ✓")
print()

# Sample: first 6 segments
print("-" * 55)
print("  SAMPLE OUTPUT (first 6 segments)")
print("-" * 55)
for seg in d["segments"][:6]:
    route = seg.get("translation_route", "?")
    print(f"  [{seg['start_fmt']}][{route[:1]}] {seg['text'][:60]}")
    print(f"                     → {seg.get('english_text','')[:60]}")
    print()

# Sample: first real Urdu segment
print("-" * 55)
print("  FIRST URDU SEGMENT TRANSLATED")
print("-" * 55)
for seg in d["segments"]:
    if seg.get("translation_route") == "TRANSLATED" and seg.get("is_urdu"):
        print(f"  [{seg['start_fmt']}] UR: {seg['text']}")
        print(f"             EN: {seg.get('english_text','')}")
        break

# Entities
print()
print("-" * 55)
print("  ENTITIES REMOVED (sample)")
print("-" * 55)
for e in d["entities_removed"][:8]:
    print(f"  {e['original']!r:30s} → {e['replaced_with']}  ({e['type']})")
