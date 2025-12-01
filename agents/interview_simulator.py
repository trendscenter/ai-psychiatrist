# agents/interview_simulator.py
from pathlib import Path
import os

class InterviewSimulator:
    """
    Pure transcript loader from a fixed file on disk.
    No LLM calls. No prompt. No mode/topic handling here.
    """

    def __init__(self, default_path: str | None = None, encoding: str = "utf-8"):
        # Fixed file path (hard-coded or via ENV); fallback to /mnt/data/transcripts.txt
        self.encoding = encoding
        self.transcript_path = (
            default_path
            or os.getenv("TRANSCRIPT_PATH")
            or "agents/transcript.txt"
        )

    def load(self) -> str:
        p = Path(self.transcript_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Transcript file not found: {self.transcript_path}")

        text = p.read_text(encoding=self.encoding).strip()
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        if not text:
            raise ValueError(f"Transcript file is empty: {self.transcript_path}")
        return text
