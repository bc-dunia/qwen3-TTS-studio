"""Parser for user-provided multi-speaker podcast scripts.

Converts free-text dialog (e.g. "Alex: Hello everyone!") into the structured
dialogue format expected by ``transcript_from_struct``.

Supported formats
-----------------
- ``Speaker Name: dialogue text``
- ``SPEAKER NAME: dialogue text``
- Multi-line continuations (lines without a colon prefix are appended to
  the previous speaker's text, joined with a single space).
- Blank lines are skipped.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Pattern: line starts with a speaker label followed by a colon and whitespace.
# Uses a broad match for the label (up to 60 non-colon chars), then validates
# separately to reject URLs, timestamps, and pure-digit labels while allowing
# Unicode speaker names (Korean, Chinese, Japanese, etc.).
_SPEAKER_LINE_RE = re.compile(
    r"^(?P<speaker>[^:\n]{1,60}):\s+(?P<text>.+)$"
)


def _is_valid_speaker_label(label: str) -> bool:
    """Return True if *label* looks like a speaker name, not a URL or timestamp."""
    stripped = label.strip()
    if not stripped:
        return False
    # Reject if first char is not a letter (any script) — blocks '10:', 'http:', etc.
    if not stripped[0].isalpha():
        return False
    # Reject labels that look like URLs.
    if stripped.startswith(("http", "ftp", "mailto")):
        return False
    # Allow letters, digits, spaces, hyphens, underscores, dots in the rest.
    for ch in stripped:
        if not (ch.isalnum() or ch in " _.-"):
            return False
    return True

MAX_SPEAKERS = 4


@dataclass
class ParseResult:
    """Result of parsing a custom podcast script."""

    dialogues: list[dict[str, str]] = field(default_factory=list)
    speakers: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.dialogues) > 0 and len(self.errors) == 0


def parse_script(text: str) -> ParseResult:
    """Parse multi-speaker script text into structured dialogues.

    Parameters
    ----------
    text:
        Raw script text pasted by the user.

    Returns
    -------
    ParseResult
        Contains ``dialogues`` (list of ``{"speaker": ..., "text": ...}``),
        ``speakers`` (unique names in order of first appearance), and
        ``errors`` (human-readable validation messages, empty when valid).
    """
    if not text or not text.strip():
        return ParseResult(errors=["Script is empty. Paste your dialog text above."])

    dialogues: list[dict[str, str]] = []
    speakers_seen: dict[str, str] = {}  # lowercase -> canonical name
    speakers_ordered: list[str] = []

    current_speaker: str | None = None
    current_text_parts: list[str] = []

    def _flush() -> None:
        """Flush the accumulated text for the current speaker into dialogues."""
        nonlocal current_speaker, current_text_parts
        if current_speaker is not None and current_text_parts:
            combined = " ".join(current_text_parts).strip()
            if combined:
                dialogues.append({"speaker": current_speaker, "text": combined})
        current_text_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _SPEAKER_LINE_RE.match(line)
        if match and _is_valid_speaker_label(match.group("speaker")):
            # Flush previous speaker's accumulated text.
            _flush()

            speaker_raw = match.group("speaker").strip()
            speaker_key = speaker_raw.lower()

            # Normalise to canonical capitalisation (first occurrence wins).
            if speaker_key not in speakers_seen:
                speakers_seen[speaker_key] = speaker_raw
                speakers_ordered.append(speaker_raw)
            current_speaker = speakers_seen[speaker_key]

            trailing_text = match.group("text").strip()
            if trailing_text:
                current_text_parts.append(trailing_text)
        else:
            # Continuation line — append to previous speaker's text.
            if current_speaker is not None:
                current_text_parts.append(line)
            # Lines before any speaker label are silently ignored.

    # Flush last speaker.
    _flush()

    # ---- validation ----
    errors: list[str] = []

    if not dialogues:
        errors.append(
            "No dialogue lines detected. Use the format 'Speaker Name: text'."
        )
        return ParseResult(errors=errors)

    if len(speakers_ordered) > MAX_SPEAKERS:
        errors.append(
            f"Too many speakers ({len(speakers_ordered)}). Maximum is {MAX_SPEAKERS}."
        )

    return ParseResult(
        dialogues=dialogues,
        speakers=speakers_ordered,
        errors=errors,
    )
