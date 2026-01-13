#!/usr/bin/env python3
"""Transcribe audio using OpenAI's Whisper API."""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def transcribe(audio_path: str):
    audio_file = Path(audio_path)
    output_file = audio_file.with_suffix(".txt")

    print(f"[INFO] Starting transcription")
    print(f"[INFO] Input: {audio_file}")
    print(f"[INFO] Size: {audio_file.stat().st_size / 1024 / 1024:.2f} MB")

    client = OpenAI()

    print(f"[INFO] Uploading to OpenAI Whisper API...")
    start_time = time.time()

    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text"
        )

    elapsed = time.time() - start_time
    print(f"[INFO] Transcription completed in {elapsed:.1f}s")

    output_file.write_text(transcript)
    print(f"[INFO] Saved to: {output_file}")
    print(f"[INFO] Output size: {len(transcript)} characters")

if __name__ == "__main__":
    audio = sys.argv[1] if len(sys.argv) > 1 else "Smart Jal orientation session.mp3"
    transcribe(audio)
