"""Voice selection and speaker profile management for podcast generation."""

import json
from pathlib import Path
from typing import Any

from podcast.models import Speaker, SpeakerProfile

SAVED_VOICES_DIR = Path("saved_voices")


def get_saved_voices() -> list[dict[str, Any]]:
    """Load saved voices from disk."""
    voices = []
    if SAVED_VOICES_DIR.exists():
        for voice_dir in SAVED_VOICES_DIR.iterdir():
            if voice_dir.is_dir():
                meta_path = voice_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        meta["id"] = voice_dir.name
                        voices.append(meta)
    return sorted(voices, key=lambda x: x.get("created", ""), reverse=True)


def get_available_voices() -> list[dict[str, Any]]:
    """
    Get all available voices (preset + saved).
    
    Returns:
        List of voice dicts with keys: voice_id, name, type, created (if saved)
    
    Note: Uses hardcoded preset voices to avoid loading the TTS model.
    Model is only loaded when actually generating audio.
    """
    voices = []
    
    # Hardcoded preset speakers - no model loading needed
    preset_speakers = ["serena", "ryan", "vivian", "aiden", "dylan", "eric", "sohee", "uncle_fu", "ono_anna"]
    for speaker in preset_speakers:
        voices.append({
            "voice_id": speaker,
            "name": speaker.replace("_", " ").title(),
            "type": "preset"
        })
    
    # Add saved voices from disk (no model loading)
    try:
        saved = get_saved_voices()
        for voice in saved:
            voices.append({
                "voice_id": voice.get("id"),
                "name": voice.get("name", voice.get("id")),
                "type": "saved",
                "created": voice.get("created")
            })
    except Exception as e:
        print(f"Warning: Could not load saved voices: {e}")
    
    return voices


def create_speaker_profile(voice_selections: list[dict[str, str]]) -> SpeakerProfile:
    """
    Create a SpeakerProfile from voice selections.
    
    Args:
        voice_selections: List of dicts with keys:
            - voice_id: str (required) - voice identifier
            - role: str (required) - Host, Expert, Guest, or Narrator
            - type: str (optional, default "preset") - preset or saved
            - name: str (optional, default to voice_id) - speaker display name
    
    Returns:
        SpeakerProfile instance
    
    Raises:
        ValueError: If validation fails (invalid voice_id, missing voice, invalid role, etc.)
    """
    if not voice_selections:
        raise ValueError("At least one voice selection is required.")
    
    if isinstance(voice_selections, dict):
        voice_selections = list(voice_selections.values())
    
    if len(voice_selections) > 4:
        raise ValueError("Maximum 4 voices allowed.")
    
    # Get available voices for validation
    available = get_available_voices()
    available_ids = {v["voice_id"] for v in available}
    
    speakers = []
    seen_roles = set()
    
    for selection in voice_selections:
        voice_id = selection.get("voice_id", "").strip()
        role = selection.get("role", "").strip()
        voice_type = selection.get("type", "preset").strip().lower()
        name = selection.get("name", voice_id).strip()
        
        # Validate voice_id
        if not voice_id:
            raise ValueError("voice_id is required for each voice selection.")
        
        if voice_id not in available_ids:
            raise ValueError(
                f"Invalid voice_id '{voice_id}'. "
                f"Available voices: {', '.join(sorted(available_ids))}"
            )
        
        # Validate role
        if not role:
            raise ValueError("role is required for each voice selection.")
        
        # Validate type
        if voice_type not in ("preset", "saved"):
            raise ValueError(f"Invalid type '{voice_type}'. Must be 'preset' or 'saved'.")
        
        # Warn about duplicate roles (not an error, but discouraged)
        if role.title() in seen_roles:
            print(f"Warning: Duplicate role '{role}' detected. Unique roles are preferred.")
        seen_roles.add(role.title())
        
        # Create Speaker instance (validation happens in Speaker class)
        speaker = Speaker(
            name=name,
            voice_id=voice_id,
            role=role,
            type=voice_type
        )
        speakers.append(speaker)
    
    # Create and return SpeakerProfile (validation happens in SpeakerProfile class)
    return SpeakerProfile(speakers=speakers)


if __name__ == "__main__":
    # Test: Get available voices
    print("=== Available Voices ===")
    voices = get_available_voices()
    for v in voices[:5]:  # Show first 5
        print(f"  {v['voice_id']} ({v['type']})")
    if len(voices) > 5:
        print(f"  ... and {len(voices) - 5} more")
    
    # Test: Create speaker profile with valid voices
    print("\n=== Test 1: Valid Profile ===")
    try:
        if len(voices) >= 2:
            selections = [
                {
                    "voice_id": voices[0]["voice_id"],
                    "role": "Host",
                    "type": voices[0]["type"],
                    "name": "Speaker A"
                },
                {
                    "voice_id": voices[1]["voice_id"],
                    "role": "Guest",
                    "type": voices[1]["type"],
                    "name": "Speaker B"
                }
            ]
            profile = create_speaker_profile(selections)
            print(f"✓ Created profile with {len(profile.speakers)} speakers")
            for speaker in profile.speakers:
                print(f"  - {speaker.name} ({speaker.role}): {speaker.voice_id}")
        else:
            print("⚠ Not enough voices available for testing")
    except Exception as e:
        print(f"✗ Error: {e}")
    
    # Test: Invalid voice_id
    print("\n=== Test 2: Invalid voice_id ===")
    try:
        selections = [
            {
                "voice_id": "nonexistent_voice",
                "role": "Host",
                "type": "preset",
                "name": "Bad Speaker"
            }
        ]
        profile = create_speaker_profile(selections)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test: Missing role
    print("\n=== Test 3: Missing role ===")
    try:
        if voices:
            selections = [
                {
                    "voice_id": voices[0]["voice_id"],
                    "type": voices[0]["type"],
                    "name": "No Role Speaker"
                }
            ]
            profile = create_speaker_profile(selections)
            print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test: Invalid role
    print("\n=== Test 4: Invalid role ===")
    try:
        if voices:
            selections = [
                {
                    "voice_id": voices[0]["voice_id"],
                    "role": "InvalidRole",
                    "type": voices[0]["type"],
                    "name": "Bad Role Speaker"
                }
            ]
            profile = create_speaker_profile(selections)
            print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    # Test: Too many voices
    print("\n=== Test 5: Too many voices ===")
    try:
        selections = [
            {
                "voice_id": voices[i % len(voices)]["voice_id"],
                "role": ["Host", "Guest", "Expert", "Narrator", "Extra"][i],
                "type": voices[i % len(voices)]["type"],
                "name": f"Speaker {i+1}"
            }
            for i in range(5)
        ]
        profile = create_speaker_profile(selections)
        print("✗ Should have raised ValueError")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}")
    
    print("\n=== All tests completed ===")
