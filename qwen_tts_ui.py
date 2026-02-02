#!/usr/bin/env python3
import os

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

import torch
import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
import os
import json
import pickle
import shutil
import zipfile
import time
import gc
import queue
import threading
import traceback
import html as html_escape
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from podcast import orchestrator as podcast_orchestrator
from ui.content_input import (
    get_content_components,
    update_topic_char_count,
    validate_content,
    get_content_dict,
)
from ui.voice_cards import (
    get_selection_summary,
    validate_selections,
    ROLES,
    MAX_VOICES,
    generate_preview,
)
from ui.progress import (
    GenerationStep,
    ProgressState,
    ProgressTracker,
    create_step_indicator_html,
    create_status_text,
    calculate_overall_progress,
    format_time_remaining,
    PROGRESS_CSS,
)
from ui.draft_preview import (
    build_outline_html,
    render_dialogues_html,
    get_segment_dialogues,
    DRAFT_PREVIEW_CSS,
)
from storage.persona_models import (
    ALLOWED_PERSONALITIES,
    ALLOWED_SPEAKING_STYLES,
    Persona,
)
from storage.persona import delete_persona, list_personas, load_persona, save_persona
from podcast.models import Outline, Transcript, SpeakerProfile
from storage.voice import get_available_voices, get_saved_voices, create_speaker_profile
from config import get_openai_api_key


def auto_transcribe_audio(audio_path: str | None) -> str:
    """
    Transcribe audio using OpenAI Whisper API.

    Args:
        audio_path: Path to the audio file to transcribe

    Returns:
        Transcribed text or error message
    """
    if not audio_path:
        return "Error: No audio file provided. Please upload or record audio first."

    if not os.path.exists(audio_path):
        return f"Error: Audio file not found at {audio_path}"

    try:
        from openai import OpenAI, APIError, APITimeoutError, RateLimitError
    except ImportError:
        return "Error: OpenAI package not installed. Please install it with: pip install openai"

    try:
        api_key = get_openai_api_key()
    except ValueError as e:
        return f"Error: {str(e)}"

    try:
        client = OpenAI(api_key=api_key)

        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        return transcript.text

    except RateLimitError:
        return (
            "Error: OpenAI API rate limit exceeded. Please wait a moment and try again."
        )
    except APITimeoutError:
        return "Error: OpenAI API request timed out. Please try again."
    except APIError as e:
        return f"Error: OpenAI API error - {str(e)}"
    except Exception as e:
        return f"Error: Failed to transcribe audio - {str(e)}"


def format_user_error(error: Exception) -> str:
    error_messages = {
        "CUDA out of memory": "Not enough GPU memory. Try reducing text length or use a smaller model.",
        "Connection refused": "Cannot connect to server. Please check if the service is running.",
        "Rate limit": "Too many requests. Please wait a moment and try again.",
        "Invalid audio": "The audio file could not be processed. Please try a different file.",
    }
    error_str = str(error)
    for key, msg in error_messages.items():
        if key.lower() in error_str.lower():
            return msg
    return f"An error occurred: {error_str[:200]}"


SAVED_VOICES_DIR = Path("saved_voices")
SAVED_VOICES_DIR.mkdir(exist_ok=True)
HISTORY_DIR = Path("generation_history")
HISTORY_DIR.mkdir(exist_ok=True)
SETTINGS_FILE = Path("tts_settings.json")
FAVORITES_FILE = Path("favorites.json")

PERSONA_CSS = """
.persona-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 1rem;
    padding: 1rem 0;
}
.persona-card {
    position: relative;
    background: #14141f;
    border: 1px solid #2a2a40;
    border-radius: 12px;
    padding: 1.25rem;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    overflow: hidden;
}
.persona-card:hover {
    background: #1a1a2e;
    border-color: #3a3a55;
    transform: translateY(-2px);
}
.persona-card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.75rem;
}
.persona-name {
    font-size: 1.1rem;
    font-weight: 600;
    color: #f0f0f5;
    letter-spacing: -0.01em;
}
.persona-voice-badge {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 6px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(80, 80, 255, 0.15);
    color: #8080ff;
    border: 1px solid rgba(80, 80, 255, 0.3);
}
.persona-traits {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
}
.persona-trait {
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    background: rgba(255, 255, 255, 0.05);
    color: #8888a0;
}
.persona-bio {
    font-size: 0.8rem;
    color: #555570;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.persona-gallery-empty {
    padding: 2rem;
    text-align: center;
    color: #555570;
    border: 1px dashed #2a2a40;
    border-radius: 12px;
}
"""

from audio.model_loader import MODEL_PATHS, get_model, loaded_models, _mps_cleanup

DEFAULT_PARAMS = {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "max_new_tokens": 2048,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0,
}

PARAM_PRESETS = {
    "fast": {
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "subtalker_temperature": 0.7,
        "subtalker_top_k": 30,
        "subtalker_top_p": 0.9,
    },
    "balanced": DEFAULT_PARAMS.copy(),
    "quality": {
        "temperature": 1.0,
        "top_k": 80,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "max_new_tokens": 4096,
        "subtalker_temperature": 1.0,
        "subtalker_top_k": 80,
        "subtalker_top_p": 1.0,
    },
}

PARAM_TOOLTIPS = {
    "temperature": "Lower = consistent pronunciation, Higher = varied intonation. Natural speech: 0.7-0.9, Precise reading: 0.3-0.5",
    "top_k": "Number of candidates for next token. Lower = stable, Higher = diverse. Recommended: 30-50",
    "top_p": "Probability-based token selection range. 1.0 = full range, lower = more certain. Recommended: 0.9-1.0",
    "repetition_penalty": "Prevents sound/word repetition. 1.0 = no penalty, higher = less repetition. Recommended: 1.0-1.1",
    "max_new_tokens": "(Auto-calculated based on text length. This setting is for reference only)",
    "subtalker_temperature": "Voice rhythm/accent control. Default recommended, adjust if needed",
    "subtalker_top_k": "Intonation diversity control. Default recommended",
    "subtalker_top_p": "Intonation selection range. Default recommended",
}

PODCAST_QUALITY_PRESETS = {
    "quick": {
        "num_segments": 2,
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 768,
        "subtalker_temperature": 0.7,
        "subtalker_top_k": 30,
        "subtalker_top_p": 0.9,
        "duration_estimate": "2-3 min",
        "tooltip": "Fast generation with 2-3 segments. Best for quick demos and testing. ~2-3 minutes total.",
    },
    "standard": {
        "num_segments": 4,
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "repetition_penalty": 1.05,
        "max_new_tokens": 1024,
        "subtalker_temperature": 0.9,
        "subtalker_top_k": 50,
        "subtalker_top_p": 1.0,
        "duration_estimate": "5-7 min",
        "tooltip": "Balanced quality and speed with 4-5 segments. Recommended for most podcasts. ~5-7 minutes total.",
    },
    "premium": {
        "num_segments": 6,
        "temperature": 1.0,
        "top_k": 80,
        "top_p": 1.0,
        "repetition_penalty": 1.1,
        "max_new_tokens": 1400,
        "subtalker_temperature": 1.0,
        "subtalker_top_k": 80,
        "subtalker_top_p": 1.0,
        "duration_estimate": "10-15 min",
        "tooltip": "High quality with 6-8 segments. Best for professional podcasts. ~10-15 minutes total.",
    },
}

MAX_CHARS = 2000
CHAR_WARNING_THRESHOLD = 1500


def _prompt_to_cpu(prompt_items):
    """Move voice_clone_prompt tensors to CPU to reduce MPS memory pressure."""
    if prompt_items is None:
        return None
    out = []
    for it in prompt_items:
        try:
            new_item = type(it)(
                ref_code=None if it.ref_code is None else it.ref_code.detach().cpu(),
                ref_spk_embedding=it.ref_spk_embedding.detach().cpu(),
                x_vector_only_mode=it.x_vector_only_mode,
                icl_mode=it.icl_mode,
                ref_text=getattr(it, "ref_text", None),
            )
            out.append(new_item)
        except Exception:
            out.append(it)
    return out


def estimate_max_tokens(
    text: str,
    tokens_per_char: float = 2.5,
    safety: float = 1.3,
    min_tokens: int = 256,
    max_cap: int = 4096,
) -> int:
    """
    Estimate appropriate max_new_tokens based on text length.

    At 12Hz TTS with ~6.5 Korean chars/sec:
    - 1 char ≈ 0.15 sec of audio
    - 1 sec of audio ≈ 12 tokens
    - So 1 char ≈ 1.8-2.5 tokens (with safety margin)
    """
    import math

    char_count = len(text)
    estimated = math.ceil(char_count * tokens_per_char * safety)
    return max(min_tokens, min(estimated, max_cap))


def load_settings():
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE) as f:
            return json.load(f)
    return DEFAULT_PARAMS.copy()


def save_settings(params):
    with open(SETTINGS_FILE, "w") as f:
        json.dump(params, f, indent=2)
    return True


def load_favorites():
    if FAVORITES_FILE.exists():
        with open(FAVORITES_FILE) as f:
            return set(json.load(f))
    return set()


def save_favorites(favorites):
    with open(FAVORITES_FILE, "w") as f:
        json.dump(list(favorites), f)


def toggle_favorite(item_id):
    favorites = load_favorites()
    if item_id in favorites:
        favorites.discard(item_id)
        status = "Removed from favorites"
    else:
        favorites.add(item_id)
        status = "Added to favorites"
    save_favorites(favorites)
    return status, format_history_for_display(), get_history_choices()


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        data, sr = sf.read(audio_path)
        return len(data) / sr
    except:
        return 0


def format_duration(seconds):
    """Format seconds as mm:ss or ss.xs."""
    if seconds >= 60:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"
    return f"{seconds:.1f}s"


def save_to_history(
    audio_path, text, voice_info, tab_type, gen_time=None, model_name=None, params=None
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    history_id = f"{timestamp}_{tab_type}"

    item_dir = HISTORY_DIR / history_id
    item_dir.mkdir(exist_ok=True)

    audio_dest = item_dir / "audio.wav"
    shutil.copy(audio_path, audio_dest)

    duration = get_audio_duration(str(audio_dest))

    meta = {
        "id": history_id,
        "text": text[:100] + "..." if len(text) > 100 else text,
        "full_text": text,
        "voice_info": voice_info,
        "tab_type": tab_type,
        "created": datetime.now().isoformat(),
        "duration": duration,
        "generation_time": gen_time,
        "model": model_name,
        "params": params or {},
    }
    with open(item_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return str(audio_dest)


def get_history_items(limit=50, search_query="", favorites_only=False):
    items = []
    favorites = load_favorites()

    # Collect items from generation_history (single voice generations)
    for item_dir in sorted(HISTORY_DIR.iterdir(), reverse=True):
        if item_dir.is_dir():
            meta_path = item_dir / "metadata.json"
            audio_path = item_dir / "audio.wav"
            if meta_path.exists() and audio_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    meta["audio_path"] = str(audio_path)
                    meta["is_favorite"] = meta["id"] in favorites

                    if search_query:
                        search_lower = search_query.lower()
                        text_match = search_lower in meta.get("full_text", "").lower()
                        voice_match = search_lower in meta.get("voice_info", "").lower()
                        if not (text_match or voice_match):
                            continue

                    if favorites_only and not meta["is_favorite"]:
                        continue

                    items.append(meta)

    # Collect items from podcasts directory (podcast generations)
    podcasts_dir = Path("podcasts")
    if podcasts_dir.exists():
        for podcast_dir in sorted(podcasts_dir.iterdir(), reverse=True):
            if podcast_dir.is_dir():
                meta_path = podcast_dir / "metadata.json"
                audio_path = podcast_dir / "final_podcast.mp3"
                if meta_path.exists() and audio_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        # Ensure podcast has type="podcast"
                        meta["type"] = "podcast"
                        meta["tab_type"] = "podcast"
                        meta["audio_path"] = str(audio_path)
                        meta["id"] = podcast_dir.name
                        meta["is_favorite"] = (
                            meta.get("id", podcast_dir.name) in favorites
                        )

                        # Load podcast artifacts
                        outline_path = podcast_dir / "outline.json"
                        transcript_path = podcast_dir / "transcript.json"
                        if outline_path.exists():
                            with open(outline_path) as f:
                                meta["outline"] = json.load(f)
                        if transcript_path.exists():
                            with open(transcript_path) as f:
                                meta["transcript"] = json.load(f)

                        if search_query:
                            search_lower = search_query.lower()
                            topic_match = search_lower in meta.get("topic", "").lower()
                            voice_match = (
                                search_lower in str(meta.get("speakers", "")).lower()
                            )
                            if not (topic_match or voice_match):
                                continue

                        if favorites_only and not meta["is_favorite"]:
                            continue

                        items.append(meta)

    # Sort all items by creation time (newest first)
    items.sort(key=lambda x: x.get("created", ""), reverse=True)

    # Return limited results
    return items[:limit]


def format_history_for_display(search_query="", favorites_only=False):
    items = get_history_items(50, search_query, favorites_only)
    if not items:
        if search_query:
            return '<div class="empty-state">No results found for your search.</div>'
        if favorites_only:
            return '<div class="empty-state">No favorites yet. Star items to save them here!</div>'
        return '<div class="empty-state">No generation history yet. Generate some audio to see it here!</div>'

    html_parts = []
    for item in items:
        created = item.get("created", "")[:19].replace("T", " ")
        tab = item.get("tab_type", "")
        item_id = item.get("id", "")
        duration = item.get("duration", 0)
        gen_time = item.get("generation_time")
        is_favorite = item.get("is_favorite", False)

        if tab == "podcast":
            icon = "🎙️"
            text_preview = item.get("topic", "Untitled Podcast")
            voice = f"{len(item.get('speakers', []))} speakers"
        else:
            icon = "🎤" if tab == "preset" else ("🎭" if tab == "clone" else "📚")
            text_preview = item.get("text", "")
            voice = item.get("voice_info", "")

        star = "⭐" if is_favorite else "☆"
        duration_str = format_duration(duration) if duration else "—"
        gen_time_str = f"{gen_time:.1f}s" if gen_time else "—"

        safe_id = html_escape.escape(str(item_id))
        safe_text = html_escape.escape(str(text_preview))
        safe_voice = html_escape.escape(str(voice))
        safe_created = html_escape.escape(str(created))

        html_parts.append(f"""
<div class="history-card" data-id="{safe_id}">
  <div class="history-card-header">
    <span class="history-icon">{icon}</span>
    <span class="history-time">{safe_created}</span>
    <span class="history-star" title="Toggle favorite">{star}</span>
  </div>
  <div class="history-text">{safe_text}</div>
  <div class="history-meta">
    <span class="history-voice">{safe_voice}</span>
    <span class="history-duration">🎵 {duration_str}</span>
    <span class="history-gentime">⏱ {gen_time_str}</span>
  </div>
  <div class="history-id">ID: {safe_id}</div>
</div>
""")

    return "".join(html_parts)


def get_history_choices():
    items = get_history_items(50)
    choices = []
    for item in items:
        created = item.get("created", "")[:16].replace("T", " ")
        tab = item.get("tab_type", "")
        duration = item.get("duration", 0)
        dur_str = (
            f"{int(duration // 60)}:{int(duration % 60):02d}"
            if duration >= 60
            else f"{duration:.0f}s"
            if duration
            else ""
        )

        if tab == "podcast":
            icon = "🎙️"
            topic = item.get("topic", "Untitled")[:25]
            speakers = len(item.get("speakers", []))
            label = f"{icon} {created} | {speakers} speakers | {dur_str} | {topic}..."
        else:
            icon = "🎤" if tab == "preset" else ("🎭" if tab == "clone" else "📚")
            text_preview = item.get("text", "")[:25]
            voice = item.get("voice_info", "").split(" ")[0][:10]
            label = f"{icon} {created} | {voice} | {dur_str} | {text_preview}..."

        value = item["id"]
        choices.append((label, value))
    return choices


def get_history_initial():
    choices = get_history_choices()
    if not choices:
        return choices, None, None, "", ""
    first_value = choices[0][1]
    audio, text, params = play_history_item_with_details(first_value)
    return choices, first_value, audio, text, params


def play_history_item(choice):
    if not choice:
        return None
    audio_path = HISTORY_DIR / choice / "audio.wav"
    if audio_path.exists():
        return str(audio_path)
    podcast_audio_path = Path("podcasts") / choice / "final_podcast.mp3"
    if podcast_audio_path.exists():
        return str(podcast_audio_path)
    return None


def play_history_item_with_details(choice):
    if not choice:
        return None, "", ""

    history_audio_path = HISTORY_DIR / choice / "audio.wav"
    history_meta_path = HISTORY_DIR / choice / "metadata.json"

    podcast_dir = Path("podcasts") / choice
    podcast_audio_path = podcast_dir / "final_podcast.mp3"
    podcast_meta_path = podcast_dir / "metadata.json"

    audio = None
    full_text = ""
    params_str = ""

    if history_meta_path.exists():
        audio = str(history_audio_path) if history_audio_path.exists() else None
        with open(history_meta_path) as f:
            meta = json.load(f)

        full_text = meta.get("full_text", "")
        params = meta.get("params", {})
        model = meta.get("model", "")
        duration = meta.get("duration", 0)
        gen_time = meta.get("generation_time", 0)
        max_tokens = params.get("max_new_tokens", 0) if params else 0

        info_parts = []
        if model:
            info_parts.append(f"Model: {model}")
        if duration:
            dur_str = (
                f"{int(duration // 60)}:{int(duration % 60):02d}"
                if duration >= 60
                else f"{duration:.1f}s"
            )
            info_parts.append(f"Duration: {dur_str}")
        if gen_time:
            info_parts.append(f"GenTime: {gen_time:.1f}s")
        if max_tokens:
            info_parts.append(f"Tokens: {max_tokens}")

        if params:
            param_names = {
                "temperature": "T",
                "top_k": "K",
                "top_p": "P",
                "repetition_penalty": "Rep",
            }
            for key, label in param_names.items():
                if key in params and params[key] is not None:
                    val = params[key]
                    info_parts.append(
                        f"{label}:{val:.2f}"
                        if isinstance(val, float)
                        else f"{label}:{val}"
                    )

        params_str = " | ".join(info_parts) if info_parts else "No info recorded"

    elif podcast_meta_path.exists():
        audio = str(podcast_audio_path) if podcast_audio_path.exists() else None
        with open(podcast_meta_path) as f:
            meta = json.load(f)

        topic = meta.get("topic", "")
        speakers = meta.get("speakers", [])
        duration = meta.get("duration", 0)

        info_parts = []
        info_parts.append(f"Topic: {topic}")
        if speakers:
            speaker_names = (
                [s.get("name", "Unknown") for s in speakers]
                if isinstance(speakers, list)
                else []
            )
            info_parts.append(f"Speakers: {', '.join(speaker_names)}")
        if duration:
            dur_str = (
                f"{int(duration // 60)}:{int(duration % 60):02d}"
                if duration >= 60
                else f"{duration:.1f}s"
            )
            info_parts.append(f"Duration: {dur_str}")

        full_text = topic
        params_str = " | ".join(info_parts) if info_parts else "No info recorded"

    return audio, full_text, params_str


def get_history_item_details(choice):
    if not choice:
        return "", "", ""

    history_meta_path = HISTORY_DIR / choice / "metadata.json"
    podcast_meta_path = Path("podcasts") / choice / "metadata.json"

    if history_meta_path.exists():
        with open(history_meta_path) as f:
            meta = json.load(f)
            params = meta.get("params", {})
            model = meta.get("model", "")

            if params:
                params_lines = [f"Model: {model}"] if model else []
                param_names = {
                    "temperature": "Temp",
                    "top_k": "Top-K",
                    "top_p": "Top-P",
                    "repetition_penalty": "Rep.Pen",
                    "max_new_tokens": "MaxTok",
                    "subtalker_temperature": "SubTemp",
                    "subtalker_top_k": "SubTop-K",
                    "subtalker_top_p": "SubTop-P",
                    "speaker": "Speaker",
                    "language": "Lang",
                }
                for key, label in param_names.items():
                    if key in params and params[key] is not None:
                        val = params[key]
                        if isinstance(val, float):
                            params_lines.append(f"{label}: {val:.2f}")
                        else:
                            params_lines.append(f"{label}: {val}")
                params_str = (
                    " | ".join(params_lines) if params_lines else "No params recorded"
                )
            else:
                params_str = "No params recorded"

            return meta.get("full_text", ""), meta.get("voice_info", ""), params_str

    elif podcast_meta_path.exists():
        with open(podcast_meta_path) as f:
            meta = json.load(f)
            topic = meta.get("topic", "")
            speakers = meta.get("speakers", [])

            speaker_names = []
            if isinstance(speakers, list):
                speaker_names = [s.get("name", "Unknown") for s in speakers]

            params_lines = [f"Topic: {topic}"]
            if speaker_names:
                params_lines.append(f"Speakers: {', '.join(speaker_names)}")

            params_str = (
                " | ".join(params_lines) if params_lines else "No info recorded"
            )
            return topic, ", ".join(speaker_names), params_str

    return "", "", ""


def apply_history_params(choice):
    if not choice:
        return tuple([gr.update()] * 8) + ("Select an item first",)

    meta_path = HISTORY_DIR / choice / "metadata.json"

    if not meta_path.exists():
        return tuple([gr.update()] * 8) + ("Item not found",)

    with open(meta_path) as f:
        meta = json.load(f)

    params = meta.get("params", {})
    if not params:
        return tuple([gr.update()] * 8) + ("No params recorded for this item",)

    return (
        params.get("temperature", gr.update()),
        params.get("top_k", gr.update()),
        params.get("top_p", gr.update()),
        params.get("repetition_penalty", gr.update()),
        params.get("max_new_tokens", gr.update()),
        params.get("subtalker_temperature", gr.update()),
        params.get("subtalker_top_k", gr.update()),
        params.get("subtalker_top_p", gr.update()),
        f"✓ Applied params from {choice[:20]}...",
    )


def delete_history_item(choice, confirm_state=False):
    if not choice:
        return "Select an item first", gr.update(), gr.update(), False

    if not confirm_state:
        gr.Warning(f"⚠️ Click Delete again to confirm deletion of '{choice}'")
        return (
            f"⚠️ Click Delete again to confirm deletion of '{choice}'",
            gr.update(),
            gr.update(),
            True,
        )

    history_item_dir = HISTORY_DIR / choice
    podcast_item_dir = Path("podcasts") / choice

    deleted = False
    if history_item_dir.exists():
        shutil.rmtree(history_item_dir)
        deleted = True

    if podcast_item_dir.exists():
        shutil.rmtree(podcast_item_dir)
        deleted = True

    if deleted:
        return (
            "Deleted",
            gr.update(choices=get_history_choices(), value=None),
            None,
            False,
        )
    return "Item not found", gr.update(), gr.update(), False


def clear_all_history():
    count = 0
    for item_dir in HISTORY_DIR.iterdir():
        if item_dir.is_dir():
            shutil.rmtree(item_dir)
            count += 1
    return (
        f"✓ Cleared {count} items",
        gr.update(choices=[], value=None),
        None,
        gr.update(value=format_history_for_display()),
    )


def export_history_to_zip():
    """Export all history items to a ZIP file."""
    items = get_history_items(100)
    if not items:
        return None, "No history to export"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"/tmp/tts_history_export_{timestamp}.zip"

    with zipfile.ZipFile(zip_path, "w") as zf:
        for item in items:
            item_id = item["id"]
            audio_path = item.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                zf.write(audio_path, f"{item_id}/audio.wav")
                meta_content = json.dumps(item, indent=2, ensure_ascii=False)
                zf.writestr(f"{item_id}/metadata.json", meta_content)

    return zip_path, f"✓ Exported {len(items)} items"


def get_podcast_history_items(limit=20):
    """Get podcast-specific history items."""
    items = []
    podcasts_dir = Path("podcasts")
    if podcasts_dir.exists():
        for podcast_dir in sorted(podcasts_dir.iterdir(), reverse=True):
            if podcast_dir.is_dir():
                meta_path = podcast_dir / "metadata.json"
                audio_path = podcast_dir / "final_podcast.mp3"
                if meta_path.exists() and audio_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        meta["id"] = podcast_dir.name
                        meta["audio_path"] = str(audio_path)
                        items.append(meta)
    return items[:limit]


def get_podcast_history_choices():
    """Get podcast history as dropdown choices."""
    items = get_podcast_history_items(20)
    choices = []
    for item in items:
        created = item.get("created", "")[:16].replace("T", " ")
        topic = item.get("topic", "Untitled")[:30]
        speakers = item.get("speakers", [])
        speaker_count = len(speakers) if speakers else 0
        duration = item.get("duration", 0)
        dur_str = (
            f"{int(duration // 60)}:{int(duration % 60):02d}"
            if duration >= 60
            else f"{duration:.0f}s"
            if duration
            else ""
        )
        label = f"🎙️ {created} | {speaker_count} voices | {dur_str} | {topic}..."
        choices.append((label, item["id"]))
    return choices


def get_podcast_history_initial():
    """Get initial podcast history state."""
    choices = get_podcast_history_choices()
    if not choices:
        return choices, None, None, ""
    first_value = choices[0][1]
    audio, metadata = load_podcast_history_item(first_value)
    return choices, first_value, audio, metadata


def load_podcast_history_item(podcast_id):
    """Load a podcast from history."""
    if not podcast_id:
        return None, ""

    podcast_dir = Path("podcasts") / podcast_id
    audio_path = podcast_dir / "final_podcast.mp3"
    meta_path = podcast_dir / "metadata.json"

    audio = str(audio_path) if audio_path.exists() else None
    metadata = ""

    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
            topic = meta.get("topic", "")
            speakers = meta.get("speakers", [])
            duration = meta.get("duration", 0)
            created = meta.get("created", "")[:19].replace("T", " ")

            speaker_names = (
                [s.get("name", "Unknown") for s in speakers]
                if isinstance(speakers, list)
                else []
            )
            dur_str = (
                f"{int(duration // 60)}:{int(duration % 60):02d}"
                if duration >= 60
                else f"{duration:.1f}s"
                if duration
                else "—"
            )

            metadata = f"Topic: {topic}\nVoices: {', '.join(speaker_names)}\nDuration: {dur_str}\nCreated: {created}"

    return audio, metadata


def delete_podcast_history_item(podcast_id, confirm_state=False):
    """Delete a podcast from history."""
    if not podcast_id:
        return "Select a podcast first", gr.update(), gr.update(), False

    if not confirm_state:
        gr.Warning(f"⚠️ Click Delete again to confirm deletion of '{podcast_id}'")
        return (
            f"⚠️ Click Delete again to confirm deletion of '{podcast_id}'",
            gr.update(),
            gr.update(),
            True,
        )

    podcast_dir = Path("podcasts") / podcast_id
    if podcast_dir.exists():
        shutil.rmtree(podcast_dir)
        return (
            "Deleted",
            gr.update(choices=get_podcast_history_choices(), value=None),
            None,
            False,
        )
    return "Podcast not found", gr.update(), gr.update(), False


def get_saved_voices():
    voices = []
    for voice_dir in SAVED_VOICES_DIR.iterdir():
        if voice_dir.is_dir():
            meta_path = voice_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                    meta["id"] = voice_dir.name
                    voices.append(meta)
    return sorted(voices, key=lambda x: x.get("created", ""), reverse=True)


def get_saved_voice_choices():
    return [v["id"] for v in get_saved_voices()]


def update_char_count(text):
    count = len(text)
    if count > MAX_CHARS:
        return f'<span class="char-count char-error">{count:,} / {MAX_CHARS:,} characters (too long)</span>'
    elif count > CHAR_WARNING_THRESHOLD:
        return f'<span class="char-count char-warning">{count:,} / {MAX_CHARS:,} characters</span>'
    else:
        return f'<span class="char-count">{count:,} / {MAX_CHARS:,} characters</span>'


def generate_custom_voice(
    text,
    model_name,
    speaker,
    language,
    instruct,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    max_new_tokens,
    sub_temp,
    sub_top_k,
    sub_top_p,
    progress=gr.Progress(),
):
    if not text.strip():
        raise gr.Error("Please enter text to generate")

    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text too long ({len(text)} chars). Maximum is {MAX_CHARS}.")

    save_settings(
        {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": sub_top_k,
            "subtalker_top_p": sub_top_p,
        }
    )

    start_time = time.time()
    char_count = len(text)
    auto_max_tokens = estimate_max_tokens(text)
    est_time = max(10, char_count * 0.15)

    wavs = None
    try:
        progress(0.1, desc="Loading model...")
        model = get_model(model_name)
        load_time = time.time() - start_time

        progress(
            0.2,
            desc=f"Model loaded ({load_time:.1f}s). Generating ~{est_time:.0f}s for {char_count} chars (max {auto_max_tokens} tokens)...",
        )

        wavs, sr = model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            instruct=instruct if instruct and instruct.strip() else None,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=auto_max_tokens,
            subtalker_temperature=sub_temp,
            subtalker_top_k=int(sub_top_k),
            subtalker_top_p=sub_top_p,
        )

        gen_time = time.time() - start_time

        progress(0.9, desc=f"Saving audio ({gen_time:.1f}s)...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wavs[0], sr)

            history_path = save_to_history(
                f.name,
                text,
                f"{speaker} ({model_name})",
                "preset",
                gen_time,
                model_name=model_name,
                params={
                    "temperature": temperature,
                    "top_k": int(top_k),
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": auto_max_tokens,
                    "subtalker_temperature": sub_temp,
                    "subtalker_top_k": int(sub_top_k),
                    "subtalker_top_p": sub_top_p,
                    "speaker": speaker,
                    "language": language,
                    "instruct": instruct if instruct and instruct.strip() else None,
                },
            )

            duration = get_audio_duration(history_path)
            status = f"Done in {gen_time:.1f}s | Duration: {format_duration(duration)} | Tokens: {auto_max_tokens}"

            progress(1.0, desc="Complete!")
            return history_path, status
    except Exception as e:
        raise gr.Error(format_user_error(e))
    finally:
        del wavs
        _mps_cleanup()


def clone_voice(
    ref_audio,
    ref_text,
    model_name,
    test_text,
    language,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    max_new_tokens,
    sub_temp,
    sub_top_k,
    sub_top_p,
    progress=gr.Progress(),
):
    if ref_audio is None:
        raise gr.Error("Please upload reference audio")
    if not ref_text.strip():
        raise gr.Error("Please enter the transcript of the reference audio")

    save_settings(
        {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": sub_top_k,
            "subtalker_top_p": sub_top_p,
        }
    )

    start_time = time.time()
    wavs = None
    voice_clone_prompt = None

    try:
        progress(0.1, desc="Loading model...")
        model = get_model(model_name)
        load_time = time.time() - start_time

        progress(0.2, desc=f"Model loaded ({load_time:.1f}s). Analyzing voice...")
        voice_clone_prompt = model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=False,
        )

        output_audio = None
        if test_text.strip():
            if len(test_text) > MAX_CHARS:
                raise gr.Error(f"Test text too long ({len(test_text)} chars)")

            char_count = len(test_text)
            auto_max_tokens = estimate_max_tokens(test_text)
            est_time = max(10, char_count * 0.15)
            progress(
                0.3,
                desc=f"Generating ~{est_time:.0f}s for {char_count} chars (max {auto_max_tokens} tokens)...",
            )

            wavs, sr = model.generate_voice_clone(
                text=test_text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                temperature=temperature,
                top_k=int(top_k),
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                max_new_tokens=auto_max_tokens,
                subtalker_temperature=sub_temp,
                subtalker_top_k=int(sub_top_k),
                subtalker_top_p=sub_top_p,
            )

            gen_time = time.time() - start_time

            progress(0.9, desc=f"Saving audio ({gen_time:.1f}s)...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wavs[0], sr)
                output_audio = save_to_history(
                    f.name,
                    test_text,
                    f"Clone ({model_name})",
                    "clone",
                    gen_time,
                    model_name=model_name,
                    params={
                        "temperature": temperature,
                        "top_k": int(top_k),
                        "top_p": top_p,
                        "repetition_penalty": repetition_penalty,
                        "max_new_tokens": auto_max_tokens,
                        "subtalker_temperature": sub_temp,
                        "subtalker_top_k": int(sub_top_k),
                        "subtalker_top_p": sub_top_p,
                        "language": language,
                    },
                )
            duration = get_audio_duration(output_audio)
            audio_info = (
                f" | Duration: {format_duration(duration)} | Tokens: {auto_max_tokens}"
            )
        else:
            gen_time = time.time() - start_time
            audio_info = ""

        prompt_temp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        cpu_prompt = _prompt_to_cpu(voice_clone_prompt)
        with open(prompt_temp.name, "wb") as f:
            pickle.dump(cpu_prompt, f)

        progress(1.0, desc="Complete!")
        status = f"Done in {gen_time:.1f}s{audio_info}. Save it below to use later."
        return output_audio, prompt_temp.name, model_name, status
    except Exception as e:
        raise gr.Error(format_user_error(e))
    finally:
        del wavs
        del voice_clone_prompt
        _mps_cleanup()


def save_cloned_voice(
    voice_name, description, style_note, ref_audio, ref_text, prompt_path, model_name
):
    if not voice_name.strip():
        gr.Warning("Please enter a name for this voice")
        return "Please enter a name for this voice", gr.update()
    if not prompt_path or not os.path.exists(prompt_path):
        gr.Warning("Clone a voice first before saving")
        return "Clone a voice first before saving", gr.update()

    try:
        # Strict sanitization: only alphanumeric, underscore, hyphen
        safe_name = "".join(c for c in voice_name if c.isalnum() or c in "_-").strip()
        if not safe_name or safe_name in (".", ".."):
            gr.Warning(
                "Invalid voice name - use only letters, numbers, underscores, hyphens"
            )
            return "Invalid voice name", gr.update()

        voice_dir = (SAVED_VOICES_DIR / safe_name).resolve()
        # Ensure we're still within SAVED_VOICES_DIR (proper path containment check)
        try:
            voice_dir.relative_to(SAVED_VOICES_DIR.resolve())
        except ValueError:
            gr.Warning("Invalid voice name")
            return "Invalid voice name", gr.update()

        voice_dir.mkdir(exist_ok=True)

        shutil.copy(prompt_path, voice_dir / "prompt.pkl")

        if ref_audio and isinstance(ref_audio, str):
            shutil.copy(ref_audio, voice_dir / "ref_audio.wav")

        metadata = {
            "name": voice_name,
            "description": description,
            "style_note": style_note,
            "ref_text": ref_text,
            "model": model_name or "1.7B-Base",
            "created": datetime.now().isoformat(),
        }
        with open(voice_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        gr.Info(f"✅ Voice '{voice_name}' saved successfully!")
        return f"Saved voice: {safe_name}", gr.update(choices=get_saved_voice_choices())
    except Exception as e:
        error_msg = format_user_error(e)
        gr.Warning(f"❌ Failed to save voice: {error_msg}")
        return f"Error: {error_msg}", gr.update()


def generate_with_saved_voice(
    text,
    saved_voice_id,
    language,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    max_new_tokens,
    sub_temp,
    sub_top_k,
    sub_top_p,
    progress=gr.Progress(),
):
    if not text.strip():
        raise gr.Error("Please enter text to generate")
    if not saved_voice_id:
        raise gr.Error("Please select a saved voice")

    if len(text) > MAX_CHARS:
        raise gr.Error(f"Text too long ({len(text)} chars). Maximum is {MAX_CHARS}.")

    save_settings(
        {
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "subtalker_temperature": sub_temp,
            "subtalker_top_k": sub_top_k,
            "subtalker_top_p": sub_top_p,
        }
    )

    start_time = time.time()
    char_count = len(text)
    auto_max_tokens = estimate_max_tokens(text)
    est_time = max(10, char_count * 0.15)
    wavs = None
    voice_clone_prompt = None

    try:
        voice_dir = SAVED_VOICES_DIR / saved_voice_id
        prompt_path = voice_dir / "prompt.pkl"
        meta_path = voice_dir / "metadata.json"

        if not prompt_path.exists():
            raise gr.Error(f"Voice not found: {saved_voice_id}")

        with open(meta_path) as f:
            meta = json.load(f)
        model_name = meta.get("model", "1.7B-Base")

        progress(0.05, desc="Loading voice profile...")

        with open(prompt_path, "rb") as f:
            voice_clone_prompt = pickle.load(f)

        progress(0.1, desc=f"Loading {model_name}...")
        model = get_model(model_name)
        load_time = time.time() - start_time

        progress(
            0.2,
            desc=f"Model loaded ({load_time:.1f}s). Generating ~{est_time:.0f}s for {char_count} chars (max {auto_max_tokens} tokens)...",
        )

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=language,
            voice_clone_prompt=voice_clone_prompt,
            temperature=temperature,
            top_k=int(top_k),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=auto_max_tokens,
            subtalker_temperature=sub_temp,
            subtalker_top_k=int(sub_top_k),
            subtalker_top_p=sub_top_p,
        )

        gen_time = time.time() - start_time

        progress(0.9, desc=f"Saving audio ({gen_time:.1f}s)...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, wavs[0], sr)
            history_path = save_to_history(
                f.name,
                text,
                f"{saved_voice_id} ({model_name})",
                "saved",
                gen_time,
                model_name=model_name,
                params={
                    "temperature": temperature,
                    "top_k": int(top_k),
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "max_new_tokens": auto_max_tokens,
                    "subtalker_temperature": sub_temp,
                    "subtalker_top_k": int(sub_top_k),
                    "subtalker_top_p": sub_top_p,
                    "language": language,
                    "saved_voice_id": saved_voice_id,
                },
            )

            duration = get_audio_duration(history_path)
            status = f"Done in {gen_time:.1f}s | Duration: {format_duration(duration)} | Tokens: {auto_max_tokens}"

            progress(1.0, desc="Complete!")
            return history_path, status
    except Exception as e:
        raise gr.Error(format_user_error(e))
    finally:
        del wavs
        del voice_clone_prompt
        _mps_cleanup()


def get_voice_details(saved_voice_id):
    if not saved_voice_id:
        return "", "", "", "", None

    voice_dir = SAVED_VOICES_DIR / saved_voice_id
    meta_path = voice_dir / "metadata.json"

    if not meta_path.exists():
        return "Not found", "", "", "", None

    with open(meta_path) as f:
        meta = json.load(f)

    ref_audio_path = voice_dir / "ref_audio.wav"
    ref_audio = str(ref_audio_path) if ref_audio_path.exists() else None

    return (
        meta.get("description", ""),
        meta.get("style_note", ""),
        meta.get("ref_text", ""),
        meta.get("model", "Unknown"),
        ref_audio,
    )


def delete_saved_voice(saved_voice_id, confirm_state):
    if not saved_voice_id:
        return "Select a voice first", gr.update(), gr.update(), False

    if not confirm_state:
        gr.Warning(f"⚠️ Click Delete again to confirm deletion of '{saved_voice_id}'")
        return (
            f"⚠️ Click Delete again to confirm deletion of '{saved_voice_id}'",
            gr.update(),
            gr.update(),
            True,
        )

    voice_dir = SAVED_VOICES_DIR / saved_voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir)
        gr.Info(f"✅ Voice '{saved_voice_id}' deleted successfully")
        return (
            f"✅ Deleted: {saved_voice_id}",
            gr.update(choices=get_saved_voice_choices(), value=None),
            gr.update(value=None),
            False,
        )
    return "Voice not found", gr.update(), gr.update(), False


def apply_preset(preset_name):
    if preset_name not in PARAM_PRESETS:
        return tuple([gr.update()] * 8) + ("Unknown preset",)

    p = PARAM_PRESETS[preset_name]
    save_settings(p)

    return (
        p["temperature"],
        p["top_k"],
        p["top_p"],
        p["repetition_penalty"],
        p["max_new_tokens"],
        p["subtalker_temperature"],
        p["subtalker_top_k"],
        p["subtalker_top_p"],
        f'<span class="save-indicator show">Current: {preset_name.title()} preset</span>',
    )


def reset_params():
    save_settings(DEFAULT_PARAMS)
    return tuple(DEFAULT_PARAMS.values()) + (
        '<span class="save-indicator show">Reset to defaults</span>',
    )


def apply_podcast_preset(preset_name):
    """Apply podcast quality preset and return updated parameters and num_segments."""
    if preset_name not in PODCAST_QUALITY_PRESETS:
        return tuple([gr.update()] * 8) + (2, "Unknown preset")

    p = PODCAST_QUALITY_PRESETS[preset_name]
    save_settings(
        {
            "temperature": p["temperature"],
            "top_k": p["top_k"],
            "top_p": p["top_p"],
            "repetition_penalty": p["repetition_penalty"],
            "max_new_tokens": p["max_new_tokens"],
            "subtalker_temperature": p["subtalker_temperature"],
            "subtalker_top_k": p["subtalker_top_k"],
            "subtalker_top_p": p["subtalker_top_p"],
        }
    )

    return (
        p["temperature"],
        p["top_k"],
        p["top_p"],
        p["repetition_penalty"],
        p["max_new_tokens"],
        p["subtalker_temperature"],
        p["subtalker_top_k"],
        p["subtalker_top_p"],
        p["num_segments"],
        f'<span class="save-indicator show">Applied {preset_name} preset ({p["duration_estimate"]})</span>',
    )


def update_podcast_preset_info(preset_name):
    if preset_name not in PODCAST_QUALITY_PRESETS:
        return gr.update(value="Unknown preset")
    p = PODCAST_QUALITY_PRESETS[preset_name]
    return gr.update(
        value=f'<div style="font-size:0.8rem;color:var(--gray-600);padding:0.5rem;background:var(--gray-100);border-radius:4px;">{p["tooltip"]}</div>'
    )


def on_param_change(*args):
    params = {
        "temperature": args[0],
        "top_k": args[1],
        "top_p": args[2],
        "repetition_penalty": args[3],
        "max_new_tokens": args[4],
        "subtalker_temperature": args[5],
        "subtalker_top_k": args[6],
        "subtalker_top_p": args[7],
    }
    save_settings(params)
    return '<span class="save-indicator show">Settings saved</span>'


def search_history(query, favorites_only):
    """Search/filter history items."""
    return format_history_for_display(query, favorites_only)


def toggle_history_favorite(choice):
    if not choice:
        return "Select an item first", gr.update(), gr.update()

    item_id = choice.split(" | ")[0]
    return toggle_favorite(item_id)


SPEAKERS = [
    "aiden",
    "dylan",
    "eric",
    "ono_anna",
    "ryan",
    "serena",
    "sohee",
    "uncle_fu",
    "vivian",
]
LANGUAGES = [
    "auto",
    "chinese",
    "english",
    "french",
    "german",
    "italian",
    "japanese",
    "korean",
    "portuguese",
    "russian",
    "spanish",
]

custom_css = """
/* ===== MONOTONE MINIMAL DESIGN ===== */
:root {
    --gray-50: #f8f9fa;
    --gray-100: #f1f3f5;
    --gray-200: #e9ecef;
    --gray-300: #dee2e6;
    --gray-400: #ced4da;
    --gray-500: #6c757d;
    --gray-600: #5a6268;
    --gray-700: #495057;
    --gray-800: #343a40;
    --gray-900: #212529;
    --white: #ffffff;
    --radius: 4px;
}

/* Hide Gradio footer and settings */
footer { display: none !important; }
.gradio-container > .wrap > .contain > footer { display: none !important; }
.settings-btn, [class*="settings"] { display: none !important; }
.built-with { display: none !important; }

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    background: var(--gray-50) !important;
}

/* ===== HEADER ===== */
.main-header {
    text-align: center;
    padding: 1.25rem 0;
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--gray-200);
}

.main-title {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--gray-800);
    margin: 0;
    letter-spacing: -0.01em;
}

.sub-title {
    color: var(--gray-600);
    font-size: 0.875rem;
    margin: 0.25rem 0 0;
}

/* ===== SECTION HEADERS ===== */
.section-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--gray-200);
}

/* ===== PARAMETERS PANEL ===== */
.params-panel {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 1rem;
}

.panel-header-compact {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--gray-300);
}

/* ===== SAVE INDICATOR ===== */
.save-indicator {
    display: inline-block;
    font-size: 0.75rem;
    color: var(--gray-600);
    opacity: 0;
    transition: opacity 0.3s ease;
    padding: 0.25rem 0.5rem;
    background: var(--gray-100);
    border-radius: var(--radius);
}

.save-indicator.show {
    opacity: 1;
    animation: fadeInOut 3s ease;
}

.param-changed {
    animation: highlight-change 1s ease;
}

@keyframes fadeInOut {
    0% { opacity: 0; }
    10% { opacity: 1; }
    80% { opacity: 1; }
    100% { opacity: 0; }
}

@keyframes highlight-change {
    0% { background: rgba(100, 100, 200, 0.2); }
    100% { background: transparent; }
}

/* ===== PRESET BUTTONS ===== */
.preset-btn-group {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
}

.preset-section {
    background: var(--gray-100);
    border: 1px solid var(--gray-200);
    border-radius: var(--radius);
    padding: 0.75rem;
    margin-bottom: 0.75rem;
}

.preset-btn-lg {
    padding: 0.5rem 0.75rem !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border-radius: var(--radius) !important;
    background: var(--white) !important;
    border: 1px solid var(--gray-300) !important;
    color: var(--gray-700) !important;
}

.preset-btn-lg:hover {
    background: var(--gray-100) !important;
    border-color: var(--gray-400) !important;
}

.preset-btn-fast, .preset-btn-quality {
    border-color: var(--gray-300) !important;
}

/* ===== CHARACTER COUNTER ===== */
.char-count {
    font-size: 0.75rem;
    color: var(--gray-600);
    padding: 0.25rem 0.5rem;
    background: var(--gray-100);
    border-radius: var(--radius);
    display: inline-block;
}

.char-count.char-warning {
    color: var(--gray-700);
    background: var(--gray-200);
}

.char-count.char-error {
    color: var(--gray-800);
    background: var(--gray-300);
    font-weight: 500;
}

/* ===== GENERATE BUTTON ===== */
.generate-btn {
    min-height: 44px !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    border-radius: var(--radius) !important;
    background: var(--gray-800) !important;
    border: none !important;
    color: var(--white) !important;
}

.generate-btn:hover {
    background: var(--gray-900) !important;
}

/* ===== HISTORY SECTION ===== */
.history-section {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--gray-200);
}

.history-header {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gray-700);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.75rem;
}

.history-list {
    max-height: 300px;
    overflow-y: auto;
}

/* ===== HISTORY ITEMS (Clickable Buttons) ===== */
.history-item-btn {
    width: 100%;
    text-align: left !important;
    padding: 0.625rem 0.75rem !important;
    margin-bottom: 0.375rem !important;
    background: var(--white) !important;
    border: 1px solid var(--gray-200) !important;
    border-radius: var(--radius) !important;
    font-size: 0.8rem !important;
    color: var(--gray-800) !important;
    cursor: pointer !important;
    transition: background 0.15s ease !important;
}

.history-item-btn:hover {
    background: var(--gray-100) !important;
    border-color: var(--gray-300) !important;
}

.history-item-btn .item-text {
    display: block;
    font-weight: 400;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.history-item-btn .item-meta {
    font-size: 0.7rem;
    color: var(--gray-500);
    margin-top: 0.25rem;
}

/* ===== EMPTY STATE ===== */
.empty-state {
    text-align: center;
    padding: 1.5rem;
    color: var(--gray-600);
    font-size: 0.8rem;
}

/* ===== TABS ===== */
.tab-nav button {
    font-weight: 500 !important;
    padding: 0.625rem 1rem !important;
    font-size: 0.85rem !important;
}

/* ===== ACCORDION ===== */
.gradio-accordion {
    border: 1px solid var(--gray-200) !important;
    border-radius: var(--radius) !important;
    margin-bottom: 0.5rem !important;
    overflow: hidden;
}

.gradio-accordion > .label-wrap {
    background: var(--gray-50) !important;
    padding: 0.5rem 0.75rem !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    border-bottom: 1px solid var(--gray-200) !important;
}

.gradio-accordion > .label-wrap:hover {
    background: var(--gray-100) !important;
}

.gradio-accordion > .wrap {
    padding: 0.75rem !important;
    background: var(--white) !important;
}

/* ===== COMPACT PARAMS ===== */
.compact-params-panel {
    font-size: 0.85rem;
}

.compact-params-panel .wrap {
    gap: 0.25rem !important;
}

.compact-params-panel input[type="range"] {
    height: 4px !important;
}

.compact-params-panel label span {
    font-size: 0.8rem !important;
    color: var(--gray-700) !important;
}

.compact-params-panel .info {
    font-size: 0.7rem !important;
    line-height: 1.3 !important;
    color: var(--gray-500) !important;
    margin-top: 2px !important;
}

.compact-slider-row {
    gap: 0.375rem !important;
}

.compact-slider-row > div {
    min-width: 0 !important;
}

/* ===== MINI BUTTONS ===== */
.mini-btn-row button {
    padding: 0.375rem 0.5rem !important;
    font-size: 0.75rem !important;
    min-height: 28px !important;
    background: var(--white) !important;
    border: 1px solid var(--gray-300) !important;
    color: var(--gray-700) !important;
}

.mini-btn-row button:hover {
    background: var(--gray-100) !important;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .main-title { font-size: 1.25rem; }
    .preset-btn-group { flex-direction: column; }
}

/* ===== HIDE VISUAL NOISE ===== */
.section-header-icon { display: none; }

/* ========================================================================
   PODCAST TAB - PREMIUM ELEVATED DESIGN
   ======================================================================== */

/* ===== PODCAST TAB CONTAINER ===== */
.podcast-tab {
    background: linear-gradient(135deg, var(--gray-50) 0%, #f0f4f8 50%, var(--gray-100) 100%);
    border-radius: 12px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}

.podcast-tab::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--gray-400), var(--gray-600), var(--gray-400));
}

/* ===== PODCAST SECTION HEADERS WITH ICONS ===== */
.podcast-section-header {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--gray-800);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
    padding-bottom: 0.625rem;
    border-bottom: 2px solid var(--gray-200);
}

.podcast-section-header .section-icon {
    font-size: 1.125rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    background: var(--gray-100);
    border-radius: 6px;
}

/* Specific section header styles */
.podcast-section-content .section-icon { background: linear-gradient(135deg, #f8f9fa, #e9ecef); }
.podcast-section-voices .section-icon { background: linear-gradient(135deg, #f1f3f5, #dee2e6); }
.podcast-section-draft .section-icon { background: linear-gradient(135deg, #e9ecef, #ced4da); }
.podcast-section-generate .section-icon { background: linear-gradient(135deg, #e9ecef, #ced4da); }
.podcast-section-output .section-icon { background: linear-gradient(135deg, #f1f3f5, #e9ecef); }

/* ===== VOICE CARDS ===== */
.voice-card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
}

.voice-card {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: 10px;
    padding: 1rem;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.voice-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--gray-300);
    transition: background 0.25s ease;
}

.voice-card:hover {
    border-color: var(--gray-400);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08), 0 2px 8px rgba(0, 0, 0, 0.04);
    transform: translateY(-2px);
}

.voice-card:hover::before {
    background: var(--gray-600);
}

.voice-card.selected {
    border-color: var(--gray-600);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.12);
}

.voice-card.selected::before {
    background: var(--gray-800);
}

.voice-card-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--gray-800);
    margin-bottom: 0.375rem;
}

.voice-card-meta {
    font-size: 0.75rem;
    color: var(--gray-500);
}

/* Voice Card Role Indicators */
.voice-card-role {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-top: 0.5rem;
}

.voice-card-role.host {
    background: var(--gray-800);
    color: var(--white);
}

.voice-card-role.guest {
    background: var(--gray-200);
    color: var(--gray-700);
}

.voice-card-role.narrator {
    background: var(--gray-100);
    color: var(--gray-600);
    border: 1px solid var(--gray-300);
}

/* ===== DRAFT PREVIEW WITH TREE INDENTATION ===== */
.draft-preview-container {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: 10px;
    padding: 1.25rem;
    max-height: 500px;
    overflow-y: auto;
}

.draft-segment {
    position: relative;
    padding-left: 1.5rem;
    margin-bottom: 1rem;
}

.draft-segment::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--gray-200);
    border-radius: 1px;
}

.draft-segment:last-child::before {
    height: 1rem;
}

.draft-segment-title {
    font-weight: 600;
    font-size: 0.85rem;
    color: var(--gray-800);
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.draft-segment-title::before {
    content: '';
    width: 8px;
    height: 8px;
    background: var(--gray-400);
    border-radius: 50%;
    margin-left: -1.75rem;
}

/* Dialogue Items */
.draft-dialogue {
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    background: var(--gray-50);
    border-radius: 8px;
    border-left: 3px solid transparent;
    transition: all 0.2s ease;
}

.draft-dialogue:hover {
    background: var(--gray-100);
}

.draft-dialogue.host {
    border-left-color: var(--gray-800);
}

.draft-dialogue.guest {
    border-left-color: var(--gray-500);
}

.draft-dialogue.narrator {
    border-left-color: var(--gray-400);
    font-style: italic;
}

/* Speaker Badges */
.speaker-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.02em;
    margin-bottom: 0.375rem;
}

.speaker-badge.host {
    background: var(--gray-800);
    color: var(--white);
}

.speaker-badge.guest {
    background: var(--gray-200);
    color: var(--gray-700);
}

.speaker-badge.narrator {
    background: transparent;
    color: var(--gray-500);
    border: 1px solid var(--gray-300);
}

.dialogue-text {
    font-size: 0.85rem;
    color: var(--gray-700);
    line-height: 1.5;
}

/* ===== PROGRESS BAR WITH SMOOTH ANIMATIONS ===== */
.podcast-progress-container {
    background: var(--white);
    border: 1px solid var(--gray-200);
    border-radius: 10px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}

.podcast-progress-bar {
    height: 6px;
    background: var(--gray-200);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 1rem;
}

.podcast-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--gray-500), var(--gray-700));
    border-radius: 3px;
    transition: width 0.6s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.podcast-progress-fill::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(
        90deg,
        transparent 0%,
        rgba(255, 255, 255, 0.3) 50%,
        transparent 100%
    );
    animation: progress-shimmer 2s infinite;
}

@keyframes progress-shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* Progress Steps */
.podcast-progress-steps {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
}

.podcast-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    text-align: center;
}

.podcast-step-indicator {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--gray-100);
    border: 2px solid var(--gray-300);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.875rem;
    margin-bottom: 0.5rem;
    transition: all 0.3s ease;
}

.podcast-step.pending .podcast-step-indicator {
    background: var(--gray-100);
    border-color: var(--gray-300);
    color: var(--gray-500);
}

.podcast-step.active .podcast-step-indicator {
    background: var(--gray-200);
    border-color: var(--gray-600);
    color: var(--gray-800);
    animation: pulse-step 1.5s infinite;
}

@keyframes pulse-step {
    0%, 100% { box-shadow: 0 0 0 0 rgba(73, 80, 87, 0.4); }
    50% { box-shadow: 0 0 0 8px rgba(73, 80, 87, 0); }
}

.podcast-step.completed .podcast-step-indicator {
    background: var(--gray-700);
    border-color: var(--gray-700);
    color: var(--white);
}

.podcast-step.completed .podcast-step-indicator::after {
    content: '✓';
    font-weight: bold;
}

.podcast-step-label {
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.podcast-step.active .podcast-step-label {
    color: var(--gray-800);
    font-weight: 600;
}

.podcast-step.completed .podcast-step-label {
    color: var(--gray-700);
}

/* ===== OUTPUT SECTION ===== */
.podcast-output-card {
    background: linear-gradient(135deg, var(--gray-100), var(--white));
    border: 1px solid var(--gray-200);
    border-radius: 10px;
    padding: 1.25rem;
    transition: all 0.3s ease;
}

.podcast-output-card:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
}

.podcast-audio-player {
    background: var(--gray-800);
    border-radius: 8px;
    padding: 1rem;
    margin-top: 0.75rem;
}

.podcast-audio-player audio {
    width: 100%;
}

/* ===== HOVER STATES FOR INTERACTIVE ELEMENTS ===== */
.podcast-tab button:not(.generate-btn) {
    transition: all 0.2s ease;
}

.podcast-tab button:not(.generate-btn):hover {
    background: var(--gray-100) !important;
    border-color: var(--gray-400) !important;
}

.podcast-tab input:focus,
.podcast-tab textarea:focus,
.podcast-tab select:focus {
    border-color: var(--gray-500) !important;
    box-shadow: 0 0 0 3px rgba(73, 80, 87, 0.1) !important;
    outline: none !important;
}

.podcast-tab .generate-btn {
    background: linear-gradient(135deg, var(--gray-800), var(--gray-900)) !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}

.podcast-tab .generate-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
}

.podcast-tab .generate-btn:active {
    transform: translateY(0);
}

/* ===== RESPONSIVE DESIGN FOR 1024px+ SCREENS ===== */
@media (min-width: 1024px) {
    .podcast-tab {
        padding: 2rem;
    }
    
    .voice-card-grid {
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 1.25rem;
    }
    
    .voice-card {
        padding: 1.25rem;
    }
    
    .draft-preview-container {
        max-height: 600px;
    }
    
    .podcast-progress-steps {
        gap: 1rem;
    }
    
    .podcast-step-indicator {
        width: 40px;
        height: 40px;
        font-size: 1rem;
    }
    
    .podcast-step-label {
        font-size: 0.75rem;
    }
    
    .podcast-section-header {
        font-size: 0.9rem;
    }
    
    .podcast-section-header .section-icon {
        width: 32px;
        height: 32px;
        font-size: 1.25rem;
    }
}

@media (min-width: 1280px) {
    .voice-card-grid {
        grid-template-columns: repeat(4, 1fr);
    }
    
    .podcast-tab {
        padding: 2.5rem;
    }
}

/* ===== PODCAST TAB SPECIFIC OVERRIDES ===== */
.podcast-tab .gradio-accordion {
    border-radius: 10px !important;
    border-color: var(--gray-200) !important;
}

.podcast-tab .gradio-accordion > .label-wrap {
    padding: 0.75rem 1rem !important;
    font-size: 0.85rem !important;
}

.podcast-tab .gradio-accordion > .wrap {
    padding: 1rem !important;
}

/* ===== PODCAST LOADING STATES ===== */
.podcast-loading-skeleton {
    background: linear-gradient(
        90deg,
        var(--gray-100) 25%,
        var(--gray-200) 50%,
        var(--gray-100) 75%
    );
    background-size: 200% 100%;
    animation: skeleton-shimmer 1.5s infinite;
    border-radius: 6px;
}

@keyframes skeleton-shimmer {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

/* ===== PODCAST CONTENT INPUT ENHANCEMENTS ===== */
.podcast-topic-input {
    background: var(--white);
    border: 2px solid var(--gray-200);
    border-radius: 10px;
    transition: border-color 0.2s ease;
}

.podcast-topic-input:focus-within {
    border-color: var(--gray-500);
}

.podcast-keypoints-list {
    list-style: none;
    padding: 0;
    margin: 0;
}

.podcast-keypoint-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.75rem;
    background: var(--gray-50);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    transition: background 0.2s ease;
}

.podcast-keypoint-item:hover {
    background: var(--gray-100);
}

.podcast-keypoint-number {
    width: 24px;
    height: 24px;
    background: var(--gray-200);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--gray-700);
    flex-shrink: 0;
}

/* ===== PODCAST TAB NAVIGATION HIGHLIGHT ===== */
.tab-nav button[aria-selected="true"].podcast-tab-btn {
    background: linear-gradient(180deg, var(--white), var(--gray-50)) !important;
    border-bottom: 2px solid var(--gray-700) !important;
    color: var(--gray-900) !important;
    font-weight: 600 !important;
}

/* ===== PODCAST EMPTY STATES ===== */
.podcast-empty-state {
    text-align: center;
    padding: 3rem 2rem;
    background: var(--gray-50);
    border: 2px dashed var(--gray-300);
    border-radius: 12px;
}

.podcast-empty-state-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    opacity: 0.5;
}

.podcast-empty-state-text {
    font-size: 0.9rem;
    color: var(--gray-600);
    max-width: 300px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ===== PODCAST SUCCESS ANIMATIONS ===== */
.podcast-success-checkmark {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    background: var(--gray-800);
    border-radius: 50%;
    color: var(--white);
    font-size: 1.5rem;
    animation: checkmark-pop 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

@keyframes checkmark-pop {
    0% { transform: scale(0); }
    50% { transform: scale(1.2); }
    100% { transform: scale(1); }
}
"""

settings = load_settings()


PRESET_VOICES = [
    {"voice_id": "aiden", "name": "Aiden"},
    {"voice_id": "dylan", "name": "Dylan"},
    {"voice_id": "eric", "name": "Eric"},
    {"voice_id": "ono_anna", "name": "Anna"},
    {"voice_id": "ryan", "name": "Ryan"},
    {"voice_id": "serena", "name": "Serena"},
    {"voice_id": "sohee", "name": "Sohee"},
    {"voice_id": "uncle_fu", "name": "Uncle Fu"},
    {"voice_id": "vivian", "name": "Vivian"},
]


def _get_podcast_voice_choices() -> list[tuple[str, str]]:
    saved = get_saved_voices()
    choices = (
        [("-- Select --", "")]
        + [(f"{v['name']} (Preset)", f"preset:{v['voice_id']}") for v in PRESET_VOICES]
        + [
            (f"{v.get('name', v.get('id'))} (Saved)", f"saved:{v.get('id')}")
            for v in saved
        ]
    )
    return choices


def _get_persona_voice_choices() -> list[tuple[str, str]]:
    saved = get_saved_voices()
    choices = [
        (f"{v['name']} (preset)", f"{v['voice_id']}|preset") for v in PRESET_VOICES
    ] + [
        (f"{v.get('name', v.get('id'))} (saved)", f"{v.get('id')}|saved") for v in saved
    ]
    return choices


def _parse_persona_voice_value(value: str) -> tuple[str, str]:
    if not value or "|" not in value:
        return "", ""
    parts = value.split("|", 1)
    return parts[0], parts[1]


def _render_persona_cards(personas: list) -> str:
    if not personas:
        return '<div class="persona-gallery-empty">No personas saved yet. Create one above!</div>'

    cards_html = []
    for voice_id, voice_type, persona in personas:
        traits_html = f"""
            <span class="persona-trait">{persona.personality}</span>
            <span class="persona-trait">{persona.speaking_style}</span>
        """
        if persona.expertise:
            for exp in persona.expertise[:2]:
                traits_html += f'<span class="persona-trait">{exp}</span>'

        bio_preview = (
            persona.bio[:100] + "..." if len(persona.bio) > 100 else persona.bio
        )

        card_html = f"""
        <div class="persona-card">
            <div class="persona-card-header">
                <span class="persona-name">{persona.character_name}</span>
                <span class="persona-voice-badge">{voice_type.upper()}</span>
            </div>
            <div class="persona-traits">{traits_html}</div>
            <div class="persona-bio">{bio_preview or "No bio"}</div>
            <div style="font-size: 0.7rem; color: #555570; margin-top: 0.5rem;">
                Voice: {voice_id}
            </div>
        </div>
        """
        cards_html.append(card_html)

    return f'<div class="persona-cards-grid">{"".join(cards_html)}</div>'


def _generate_persona_voice_preview(voice_id: str, voice_type: str) -> str | None:
    try:
        from ui.voice_cards import generate_preview

        return generate_preview(voice_id, voice_type)
    except Exception as e:
        print(f"Voice preview generation failed: {format_user_error(e)}")
        return None


with gr.Blocks(title="Qwen3-TTS Studio") as demo:
    gr.HTML("""
    <div class="main-header">
        <h1 class="main-title">Qwen3-TTS Studio</h1>
        <p class="sub-title">Voice Cloning & Text-to-Speech</p>
    </div>
    """)

    current_prompt_data = gr.State(None)
    current_clone_model = gr.State(None)

    with gr.Row():
        with gr.Column(scale=5):
            with gr.Tabs() as tabs:
                with gr.TabItem("Preset Voices", id="preset"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Voice Settings</div>')

                            cv_model = gr.Radio(
                                ["1.7B-CustomVoice", "0.6B-CustomVoice"],
                                value="1.7B-CustomVoice",
                                label="Model",
                                info="1.7B: Higher quality | 0.6B: Faster",
                            )
                            cv_speaker = gr.Dropdown(
                                choices=SPEAKERS,
                                value="serena",
                                label="Voice Preset",
                                info="Select a built-in voice character",
                            )
                            cv_language = gr.Dropdown(
                                choices=LANGUAGES,
                                value="auto",
                                label="Language",
                                info="Auto-detect or specify language",
                            )
                            cv_instruct = gr.Textbox(
                                label="Voice Style (1.7B only)",
                                placeholder="e.g., Speak warmly and enthusiastically",
                                lines=2,
                                info="Optional instruction to guide voice style",
                            )

                        with gr.Column(scale=2):
                            gr.HTML('<div class="section-header">Text Input</div>')

                            cv_text = gr.Textbox(
                                label="Text to Speak",
                                placeholder="Enter the text you want to convert to speech...",
                                lines=4,
                                max_lines=8,
                            )
                            cv_char_count = gr.HTML(value=update_char_count(""))

                            cv_btn = gr.Button(
                                "Generate Speech",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                            cv_status = gr.Textbox(
                                label="Status", interactive=False, show_label=True
                            )

                            cv_audio = gr.Audio(
                                label="Generated Audio", type="filepath"
                            )
                            cv_download = gr.File(label="Download Audio", visible=False)

                            gr.HTML(
                                '<div class="history-section"><div class="history-header">Recent History</div></div>'
                            )
                            with gr.Row():
                                cv_history_search = gr.Textbox(
                                    placeholder="Search history...",
                                    show_label=False,
                                    scale=3,
                                )
                                cv_history_favorites = gr.Checkbox(
                                    label="Favorites only",
                                    value=False,
                                    scale=1,
                                )
                            cv_history_display = gr.HTML(
                                value=format_history_for_display(),
                                elem_classes=["history-display"],
                            )
                            cv_init = get_history_initial()
                            cv_history_dropdown = gr.Dropdown(
                                choices=cv_init[0],
                                value=cv_init[1],
                                label="Select to play",
                                allow_custom_value=False,
                            )
                            cv_history_text = gr.Textbox(
                                label="Text",
                                lines=2,
                                interactive=False,
                                value=cv_init[3],
                            )
                            cv_history_params = gr.Textbox(
                                label="Generation Settings",
                                lines=1,
                                interactive=False,
                                value=cv_init[4],
                            )
                            cv_history_audio = gr.Audio(
                                label="Playback",
                                type="filepath",
                                interactive=False,
                                value=cv_init[2],
                            )
                            with gr.Row(elem_classes=["mini-btn-row"]):
                                cv_history_refresh = gr.Button("Refresh", size="sm")
                                cv_history_apply = gr.Button(
                                    "Apply Settings", size="sm"
                                )
                                cv_history_favorite = gr.Button("★ Favorite", size="sm")
                                cv_history_delete = gr.Button(
                                    "Delete", size="sm", variant="stop"
                                )
                            cv_history_delete_confirm = gr.State(False)

                with gr.TabItem("Clone Voice", id="clone"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Reference Audio</div>')

                            vc_ref_audio = gr.Audio(
                                label="Upload Reference Audio",
                                type="filepath",
                                sources=["upload", "microphone"],
                            )
                            vc_ref_text = gr.Textbox(
                                label="Reference Transcript",
                                placeholder="Enter the exact words spoken in the reference audio...",
                                lines=3,
                                info="Must match the reference audio exactly",
                            )
                            with gr.Row():
                                vc_auto_transcribe_btn = gr.Button(
                                    "Auto-transcribe", size="sm"
                                )
                            vc_model = gr.Radio(
                                ["1.7B-Base", "0.6B-Base"],
                                value="1.7B-Base",
                                label="Model",
                            )
                            vc_language = gr.Dropdown(
                                choices=LANGUAGES, value="auto", label="Output Language"
                            )

                        with gr.Column(scale=2):
                            gr.HTML(
                                '<div class="section-header">Test Cloned Voice</div>'
                            )

                            vc_test_text = gr.Textbox(
                                label="Test Text",
                                placeholder="Enter text to test the cloned voice...",
                                lines=3,
                                info="Leave empty to just create the voice profile",
                            )
                            vc_test_char_count = gr.HTML(value=update_char_count(""))

                            vc_clone_btn = gr.Button(
                                "Clone & Generate",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                            vc_status = gr.Textbox(label="Status", interactive=False)
                            vc_output = gr.Audio(label="Test Output", type="filepath")
                            vc_download = gr.File(label="Download Audio", visible=False)

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Save Cloned Voice</div>'
                            )

                            with gr.Row():
                                vc_name = gr.Textbox(
                                    label="Voice Name",
                                    placeholder="my_custom_voice",
                                    scale=2,
                                )
                                vc_save_btn = gr.Button("Save Voice", scale=1)

                            vc_description = gr.Textbox(
                                label="Description",
                                placeholder="Deep male voice with British accent...",
                                lines=2,
                            )
                            vc_style_note = gr.Textbox(
                                label="Usage Notes",
                                placeholder="Best for narration, avoid singing...",
                                lines=1,
                            )
                            vc_save_status = gr.Textbox(
                                label="", interactive=False, show_label=False
                            )

                with gr.TabItem("Saved Voices", id="saved"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Your Voices</div>')

                            sv_voice_dropdown = gr.Dropdown(
                                choices=get_saved_voice_choices(),
                                label="Select Voice",
                                info="Choose from your saved voice clones",
                            )

                            with gr.Row():
                                sv_refresh_btn = gr.Button("Refresh", size="sm")
                                sv_delete_btn = gr.Button(
                                    "Delete", variant="stop", size="sm"
                                )

                            sv_model_info = gr.Textbox(
                                label="Model Used", interactive=False
                            )
                            sv_description = gr.Textbox(
                                label="Description", interactive=False, lines=2
                            )
                            sv_ref_audio = gr.Audio(
                                label="Original Reference",
                                type="filepath",
                                interactive=False,
                            )

                        with gr.Column(scale=2):
                            gr.HTML(
                                '<div class="section-header">Generate with Saved Voice</div>'
                            )

                            sv_text = gr.Textbox(
                                label="Text to Speak",
                                placeholder="Enter text to generate with the saved voice...",
                                lines=4,
                            )
                            sv_char_count = gr.HTML(value=update_char_count(""))

                            sv_language = gr.Dropdown(
                                choices=LANGUAGES, value="auto", label="Language"
                            )

                            sv_generate_btn = gr.Button(
                                "Generate Speech",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                            sv_status = gr.Textbox(label="Status", interactive=False)
                            sv_audio = gr.Audio(
                                label="Generated Audio", type="filepath"
                            )
                            sv_download = gr.File(label="Download Audio", visible=False)

                            gr.HTML(
                                '<div class="history-section"><div class="history-header">Recent History</div></div>'
                            )
                            with gr.Row():
                                sv_history_search = gr.Textbox(
                                    placeholder="Search history...",
                                    show_label=False,
                                    scale=3,
                                )
                                sv_history_favorites = gr.Checkbox(
                                    label="Favorites only",
                                    value=False,
                                    scale=1,
                                )
                            sv_history_display = gr.HTML(
                                value=format_history_for_display(),
                                elem_classes=["history-display"],
                            )
                            sv_init = get_history_initial()
                            sv_history_dropdown = gr.Dropdown(
                                choices=sv_init[0],
                                value=sv_init[1],
                                label="Select to play",
                                allow_custom_value=False,
                            )
                            sv_history_text = gr.Textbox(
                                label="Text",
                                lines=2,
                                interactive=False,
                                value=sv_init[3],
                            )
                            sv_history_params = gr.Textbox(
                                label="Generation Settings",
                                lines=1,
                                interactive=False,
                                value=sv_init[4],
                            )
                            sv_history_audio = gr.Audio(
                                label="Playback",
                                type="filepath",
                                interactive=False,
                                value=sv_init[2],
                            )
                            with gr.Row(elem_classes=["mini-btn-row"]):
                                sv_history_refresh = gr.Button("Refresh", size="sm")
                                sv_history_apply = gr.Button(
                                    "Apply Settings", size="sm"
                                )
                                sv_history_favorite = gr.Button("★ Favorite", size="sm")
                                sv_history_delete = gr.Button(
                                    "Delete", size="sm", variant="stop"
                                )
                            sv_history_delete_confirm = gr.State(False)

                    sv_style_note = gr.Textbox(visible=False)
                    sv_ref_text = gr.Textbox(visible=False)
                    sv_delete_status = gr.Textbox(visible=False)
                    sv_delete_confirm = gr.State(False)

                with gr.TabItem("Personas", id="personas"):
                    gr.HTML(f"<style>{PERSONA_CSS}</style>")

                    gr.Markdown("## Persona Management")
                    gr.Markdown("*Define character personas for your podcast voices*")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Voice Selection</div>')

                            persona_voice_dropdown = gr.Dropdown(
                                label="Select Voice",
                                choices=_get_persona_voice_choices(),
                                value=None,
                                interactive=True,
                                allow_custom_value=False,
                                info="Choose a voice to create or edit its persona",
                            )

                            persona_refresh_voices_btn = gr.Button(
                                "Refresh Voices", size="sm"
                            )

                        with gr.Column(scale=2):
                            gr.HTML(
                                '<div class="section-header">Character Definition</div>'
                            )

                            persona_character_name = gr.Textbox(
                                label="Character Name",
                                placeholder="e.g., Dr. Sarah Chen, The Wise Narrator",
                                info="Display name for this character",
                            )

                            with gr.Row():
                                persona_personality = gr.Dropdown(
                                    label="Personality",
                                    choices=sorted(ALLOWED_PERSONALITIES),
                                    value=None,
                                    interactive=True,
                                    info="Core personality trait",
                                )
                                persona_speaking_style = gr.Dropdown(
                                    label="Speaking Style",
                                    choices=sorted(ALLOWED_SPEAKING_STYLES),
                                    value=None,
                                    interactive=True,
                                    info="How they communicate",
                                )

                            persona_expertise = gr.Textbox(
                                label="Expertise (comma-separated)",
                                placeholder="e.g., AI Ethics, Philosophy, Technology",
                                info="Areas of knowledge or expertise",
                            )

                            persona_background = gr.Textbox(
                                label="Background",
                                placeholder="Brief background information about the character...",
                                lines=2,
                                info="Character's history, role, or context",
                            )

                            persona_bio = gr.Textbox(
                                label="Bio / Personality Notes",
                                placeholder="Detailed character description, personality quirks, mannerisms...",
                                lines=3,
                                info="Extended character description for transcript generation",
                            )

                            with gr.Row():
                                persona_save_btn = gr.Button(
                                    "Save Persona", variant="primary", size="lg"
                                )
                                persona_delete_btn = gr.Button(
                                    "Delete Persona", variant="stop", size="lg"
                                )
                                persona_preview_btn = gr.Button(
                                    "Preview Voice", size="lg"
                                )

                            persona_status_text = gr.Textbox(
                                label="Status", interactive=False, show_label=True
                            )

                            persona_preview_audio = gr.Audio(
                                label="Voice Preview", type="filepath", visible=True
                            )

                    gr.HTML(
                        '<div class="section-header" style="margin-top: 2rem;">Saved Personas Gallery</div>'
                    )

                    personas_gallery = gr.HTML(
                        value=_render_persona_cards(list_personas())
                    )

                    persona_refresh_gallery_btn = gr.Button(
                        "Refresh Gallery", size="sm"
                    )

                    persona_selected_voice_state = gr.State(value=None)
                    persona_delete_confirm_state = gr.State(value=False)

                    def on_persona_voice_select(voice_value: str):
                        if not voice_value:
                            return (
                                "",
                                "",
                                None,
                                None,
                                "",
                                "",
                                "Select a voice to begin",
                                voice_value,
                                False,
                            )

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            return (
                                "",
                                "",
                                None,
                                None,
                                "",
                                "",
                                "Invalid voice selection",
                                voice_value,
                                False,
                            )

                        existing = load_persona(voice_id, voice_type)

                        if existing:
                            expertise_str = (
                                ", ".join(existing.expertise)
                                if existing.expertise
                                else ""
                            )
                            return (
                                existing.character_name,
                                expertise_str,
                                existing.personality,
                                existing.speaking_style,
                                existing.background,
                                existing.bio,
                                f"Loaded persona for {voice_id}",
                                voice_value,
                                False,
                            )
                        else:
                            return (
                                "",
                                "",
                                None,
                                None,
                                "",
                                "",
                                f"No persona found for {voice_id}. Create one!",
                                voice_value,
                                False,
                            )

                    def on_persona_save(
                        voice_value, char_name, pers, style, exp, bg, bio_text
                    ):
                        if not voice_value:
                            gr.Warning("Please select a voice first")
                            return "Error: No voice selected", _render_persona_cards(
                                list_personas()
                            )

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            gr.Warning("Invalid voice selection")
                            return "Error: Invalid voice", _render_persona_cards(
                                list_personas()
                            )

                        if not char_name or not char_name.strip():
                            gr.Warning("Character name is required")
                            return (
                                "Error: Character name required",
                                _render_persona_cards(list_personas()),
                            )

                        if not pers:
                            gr.Warning("Personality is required")
                            return "Error: Personality required", _render_persona_cards(
                                list_personas()
                            )

                        if not style:
                            gr.Warning("Speaking style is required")
                            return (
                                "Error: Speaking style required",
                                _render_persona_cards(list_personas()),
                            )

                        expertise_list = (
                            [e.strip() for e in exp.split(",") if e.strip()]
                            if exp
                            else []
                        )

                        try:
                            persona = Persona(
                                voice_id=voice_id,
                                voice_type=voice_type,
                                character_name=char_name.strip(),
                                personality=pers,
                                speaking_style=style,
                                expertise=expertise_list,
                                background=bg.strip() if bg else "",
                                bio=bio_text.strip() if bio_text else "",
                            )

                            save_persona(persona)
                            gr.Info(f"Persona saved for {char_name}")
                            return f"Saved persona: {char_name}", _render_persona_cards(
                                list_personas()
                            )

                        except ValueError as e:
                            error_msg = format_user_error(e)
                            gr.Warning(f"Validation error: {error_msg}")
                            return f"Error: {error_msg}", _render_persona_cards(
                                list_personas()
                            )
                        except Exception as e:
                            error_msg = format_user_error(e)
                            gr.Warning(f"Failed to save: {error_msg}")
                            return (
                                f"Error saving persona: {error_msg}",
                                _render_persona_cards(list_personas()),
                            )

                    def on_persona_delete(voice_value, confirm_state):
                        gallery_html = _render_persona_cards(list_personas())

                        if not voice_value:
                            gr.Warning("Please select a voice first")
                            return (
                                "Error: No voice selected",
                                gallery_html,
                                "",
                                None,
                                None,
                                "",
                                "",
                                "",
                                False,
                            )

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            gr.Warning("Invalid voice selection")
                            return (
                                "Error: Invalid voice",
                                gallery_html,
                                "",
                                None,
                                None,
                                "",
                                "",
                                "",
                                False,
                            )

                        if not confirm_state:
                            gr.Warning(
                                f"Click Delete again to confirm deletion of persona for '{voice_id}'"
                            )
                            return (
                                f"Click Delete again to confirm deletion for {voice_id}",
                                gallery_html,
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                True,
                            )

                        try:
                            deleted = delete_persona(voice_id, voice_type)

                            if deleted:
                                gr.Info(f"Persona deleted for {voice_id}")
                                return (
                                    f"Deleted persona for {voice_id}",
                                    _render_persona_cards(list_personas()),
                                    "",
                                    None,
                                    None,
                                    "",
                                    "",
                                    "",
                                    False,
                                )
                            else:
                                gr.Warning(f"No persona found for {voice_id}")
                                return (
                                    f"No persona found for {voice_id}",
                                    gallery_html,
                                    "",
                                    None,
                                    None,
                                    "",
                                    "",
                                    "",
                                    False,
                                )

                        except Exception as e:
                            error_msg = format_user_error(e)
                            gr.Warning(f"Failed to delete: {error_msg}")
                            return (
                                f"Error deleting persona: {error_msg}",
                                gallery_html,
                                "",
                                None,
                                None,
                                "",
                                "",
                                "",
                                False,
                            )

                    def on_persona_preview_voice(voice_value):
                        if not voice_value:
                            gr.Warning("Please select a voice first")
                            return None, "Select a voice to preview"

                        voice_id, voice_type = _parse_persona_voice_value(voice_value)

                        if not voice_id:
                            gr.Warning("Invalid voice selection")
                            return None, "Invalid voice selection"

                        audio_path = _generate_persona_voice_preview(
                            voice_id, voice_type
                        )

                        if audio_path:
                            gr.Info("Voice preview generated!")
                            return audio_path, f"Preview generated for {voice_id}"
                        else:
                            gr.Warning("Failed to generate preview")
                            return None, "Failed to generate voice preview"

                    def on_persona_refresh_voices():
                        choices = _get_persona_voice_choices()
                        return gr.update(choices=choices, value=None)

                    def on_persona_refresh_gallery():
                        return _render_persona_cards(list_personas())

                    persona_voice_dropdown.change(
                        fn=on_persona_voice_select,
                        inputs=[persona_voice_dropdown],
                        outputs=[
                            persona_character_name,
                            persona_expertise,
                            persona_personality,
                            persona_speaking_style,
                            persona_background,
                            persona_bio,
                            persona_status_text,
                            persona_selected_voice_state,
                            persona_delete_confirm_state,
                        ],
                    )

                    persona_save_btn.click(
                        fn=on_persona_save,
                        inputs=[
                            persona_voice_dropdown,
                            persona_character_name,
                            persona_personality,
                            persona_speaking_style,
                            persona_expertise,
                            persona_background,
                            persona_bio,
                        ],
                        outputs=[persona_status_text, personas_gallery],
                    )

                    persona_delete_btn.click(
                        fn=on_persona_delete,
                        inputs=[persona_voice_dropdown, persona_delete_confirm_state],
                        outputs=[
                            persona_status_text,
                            personas_gallery,
                            persona_character_name,
                            persona_personality,
                            persona_speaking_style,
                            persona_expertise,
                            persona_background,
                            persona_bio,
                            persona_delete_confirm_state,
                        ],
                    )

                    persona_preview_btn.click(
                        fn=on_persona_preview_voice,
                        inputs=[persona_voice_dropdown],
                        outputs=[persona_preview_audio, persona_status_text],
                    )

                    persona_refresh_voices_btn.click(
                        fn=on_persona_refresh_voices, outputs=[persona_voice_dropdown]
                    )

                    persona_refresh_gallery_btn.click(
                        fn=on_persona_refresh_gallery, outputs=[personas_gallery]
                    )

                with gr.TabItem("Podcast", id="podcast"):
                    podcast_outline_state = gr.State(None)
                    podcast_transcript_state = gr.State(None)
                    podcast_speaker_profile_state = gr.State(None)
                    podcast_voice_selections_state = gr.State({})
                    podcast_session_state = gr.State(
                        None
                    )  # Stores {podcast_dir, quality_preset, language}

                    gr.HTML(f"<style>{PROGRESS_CSS}</style>")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML('<div class="section-header">Topic & Style</div>')

                            podcast_topic = gr.Textbox(
                                label="Podcast Topic",
                                placeholder="Enter your podcast topic or main subject...\n\nExample: The future of artificial intelligence",
                                lines=3,
                                max_lines=3,
                                info="Required. What is your podcast about?",
                            )
                            podcast_topic_chars = gr.HTML(
                                value=update_topic_char_count("")
                            )

                            podcast_key_points = gr.Textbox(
                                label="Key Points (Optional)",
                                placeholder="List the main points you want to cover...\n\n- Point 1\n- Point 2\n- Point 3",
                                lines=4,
                                info="Bullet points or key topics to discuss",
                            )

                            podcast_briefing = gr.Textbox(
                                label="Style & Tone (Optional)",
                                placeholder="Describe the desired style and tone...\n\nExample: Conversational and engaging",
                                lines=2,
                                info="How should the podcast sound?",
                            )

                            with gr.Row():
                                podcast_quality_preset = gr.Dropdown(
                                    choices=list(PODCAST_QUALITY_PRESETS.keys()),
                                    value="standard",
                                    label="Quality",
                                    info="Select quality level",
                                    scale=1,
                                )
                                podcast_num_segments = gr.Slider(
                                    minimum=2,
                                    maximum=8,
                                    value=4,
                                    step=1,
                                    label="Segments",
                                    info="Outline segments",
                                    scale=1,
                                )

                            podcast_language = gr.Dropdown(
                                label="Language",
                                choices=[
                                    "English",
                                    "Korean",
                                    "Chinese",
                                    "Japanese",
                                    "Spanish",
                                    "French",
                                    "German",
                                ],
                                value="English",
                                info="Language for script generation and voice synthesis",
                            )

                            with gr.Row():
                                gr.HTML(
                                    '<div class="section-header" style="margin-top:1rem;">Speakers (1-4)</div>'
                                )
                                podcast_refresh_voices_btn = gr.Button(
                                    "🔄 Refresh", size="sm", scale=0, min_width=80
                                )

                            podcast_voice_summary = gr.HTML(
                                value='<div style="color:#888; font-size:0.9em;">Select 1-4 speakers below</div>'
                            )

                            podcast_speaker_slots = []
                            podcast_preview_buttons = []
                            _slot_roles = ["Host", "Guest", "Guest", "Guest"]
                            _initial_voice_choices = (
                                _get_podcast_voice_choices()
                            )  # Cache once
                            for i in range(4):
                                with gr.Row():
                                    slot_role = gr.Dropdown(
                                        choices=ROLES,
                                        value=_slot_roles[i],
                                        label=f"Speaker {i + 1}",
                                        scale=1,
                                        min_width=100,
                                        interactive=True,
                                    )
                                    slot_voice = gr.Dropdown(
                                        choices=_initial_voice_choices,
                                        value="",
                                        label="Voice",
                                        scale=2,
                                        min_width=150,
                                        interactive=True,
                                        allow_custom_value=False,
                                    )
                                    slot_preview = gr.Button(
                                        "▶",
                                        size="sm",
                                        scale=0,
                                        min_width=40,
                                    )
                                    podcast_preview_buttons.append(slot_preview)
                                    podcast_speaker_slots.append(
                                        (slot_role, slot_voice)
                                    )

                            podcast_preview_audio = gr.Audio(
                                label="Preview",
                                visible=True,
                                interactive=False,
                            )

                            podcast_voice_status = gr.HTML(value="")

                            podcast_generate_btn = gr.Button(
                                "Generate Podcast",
                                variant="primary",
                                elem_classes=["generate-btn"],
                                size="lg",
                            )

                        with gr.Column(scale=2):
                            gr.HTML('<div class="section-header">Progress</div>')

                            podcast_step_indicator = gr.HTML(
                                value=create_step_indicator_html(
                                    GenerationStep.OUTLINE, 0.0
                                )
                            )

                            podcast_overall_progress = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                label="Overall Progress",
                                interactive=False,
                            )

                            podcast_status = gr.Textbox(
                                value="Ready to generate...",
                                label="Status",
                                interactive=False,
                            )

                            podcast_time_remaining = gr.Textbox(
                                value="", label="", interactive=False, show_label=False
                            )

                            podcast_error_display = gr.HTML(value="")

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Draft Preview</div>'
                            )

                            with gr.Row():
                                with gr.Column(scale=1):
                                    gr.HTML(
                                        '<div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.85rem;">Outline</div>'
                                    )
                                    podcast_outline_html = gr.HTML(
                                        value='<div class="empty-state">Generate a podcast to see the outline</div>'
                                    )

                                with gr.Column(scale=2):
                                    gr.HTML(
                                        '<div style="font-weight: 600; margin-bottom: 0.5rem; font-size: 0.85rem;">Transcript</div>'
                                    )
                                    podcast_transcript_html = gr.HTML(
                                        value='<div class="empty-state">Generate a podcast to see the transcript</div>'
                                    )

                            with gr.Accordion(
                                "Edit Transcript", open=False, visible=False
                            ) as podcast_edit_accordion:
                                gr.HTML(
                                    '<div style="font-size: 0.8rem; color: var(--gray-600); margin-bottom: 0.5rem;">'
                                    'Edit dialogue text below, then click "Regenerate Audio" to apply changes.</div>'
                                )
                                podcast_transcript_editor = gr.Dataframe(
                                    headers=["Speaker", "Text"],
                                    datatype=["str", "str"],
                                    interactive=True,
                                    wrap=True,
                                    value=[],
                                )
                                podcast_regenerate_btn = gr.Button(
                                    "Regenerate Audio from Edits",
                                    variant="primary",
                                    size="sm",
                                )

                            gr.HTML(
                                '<div class="section-header" style="margin-top:1rem;">Audio Output</div>'
                            )

                            podcast_final_audio = gr.Audio(
                                label="Generated Podcast",
                                type="filepath",
                                interactive=False,
                            )

                            podcast_download = gr.File(
                                label="Download Podcast", visible=False
                            )

                            gr.HTML(
                                '<div class="history-section"><div class="history-header">Podcast History</div></div>'
                            )
                            with gr.Row():
                                podcast_history_search = gr.Textbox(
                                    placeholder="Search history...",
                                    show_label=False,
                                    scale=3,
                                )
                                podcast_history_favorites = gr.Checkbox(
                                    label="Favorites only",
                                    value=False,
                                    scale=1,
                                )
                            podcast_history_display = gr.HTML(
                                value=format_history_for_display(),
                                elem_classes=["history-display"],
                            )
                            podcast_hist_init = get_podcast_history_initial()
                            podcast_history_dropdown = gr.Dropdown(
                                choices=podcast_hist_init[0],
                                value=podcast_hist_init[1],
                                label="Select to load",
                                allow_custom_value=False,
                            )
                            podcast_history_metadata = gr.Textbox(
                                label="Details",
                                lines=3,
                                interactive=False,
                                value=podcast_hist_init[3],
                            )
                            podcast_history_audio = gr.Audio(
                                label="Playback",
                                type="filepath",
                                interactive=False,
                                value=podcast_hist_init[2],
                            )
                            with gr.Row(elem_classes=["mini-btn-row"]):
                                podcast_history_refresh = gr.Button(
                                    "Refresh", size="sm"
                                )
                                podcast_history_favorite = gr.Button(
                                    "★ Favorite", size="sm"
                                )
                                podcast_history_delete = gr.Button(
                                    "Delete", size="sm", variant="stop"
                                )
                            podcast_history_delete_confirm = gr.State(False)

                    def build_voice_selections_from_slots(*slot_values):
                        sels = {}
                        num_slots = len(slot_values) // 2
                        for i in range(num_slots):
                            role = slot_values[i * 2]
                            voice_val = slot_values[i * 2 + 1]
                            if voice_val and voice_val != "":
                                parts = voice_val.split(":", 1)
                                if len(parts) != 2:
                                    summary = (
                                        '<div style="color:#dc3545;">'
                                        "Invalid voice selection. Choose a preset or saved voice."
                                        "</div>"
                                    )
                                    return {}, summary
                                vtype, vid = parts
                                if vtype not in {"preset", "saved"} or not vid:
                                    summary = (
                                        '<div style="color:#dc3545;">'
                                        "Invalid voice selection. Choose a preset or saved voice."
                                        "</div>"
                                    )
                                    return {}, summary
                                sels[f"slot_{i}"] = {
                                    "voice_id": vid,
                                    "name": vid,
                                    "role": role,
                                    "type": vtype,
                                }
                        count = len(sels)
                        if count == 0:
                            summary = (
                                '<div style="color:#888;">Select 1-4 speakers</div>'
                            )
                        elif count == 1:
                            summary = '<div style="color:#28a745;">1 speaker selected ✓ (Narration mode)</div>'
                        else:
                            summary = f'<div style="color:#28a745;">{count} speakers selected ✓</div>'
                        return sels, summary

                    def play_podcast_preview(voice_value):
                        """Generate and play preview for selected voice."""
                        if not voice_value or voice_value.strip() == "":
                            return None

                        try:
                            # Parse voice_value format: "preset:serena" or "saved:my_voice"
                            parts = voice_value.split(":", 1)
                            if len(parts) != 2:
                                return None

                            voice_type, voice_id = parts
                            audio_path = generate_preview(voice_id, voice_type)

                            if audio_path and os.path.exists(audio_path):
                                return audio_path
                            return None
                        except Exception as e:
                            print(f"Error playing preview: {e}")
                            return None

                    def validate_podcast_voices(selections):
                        is_valid, message, _ = validate_selections(selections)
                        if is_valid:
                            return f'<div style="color: #28a745;">{message}</div>'
                        return f'<div style="color: #dc3545;">{message}</div>'

                    @dataclass
                    class _ProgressEvent:
                        step: GenerationStep
                        progress: float
                        status: str
                        detail: str
                        data: dict | None = None

                    @dataclass
                    class _DoneEvent:
                        result: dict[str, Any]

                    @dataclass
                    class _ErrorEvent:
                        error: str
                        tb: str

                    def run_podcast_generation(
                        topic,
                        key_points,
                        briefing,
                        num_segments,
                        voice_selections,
                        quality_preset,
                        language,
                    ):
                        is_valid, error_msg = validate_content(
                            topic, key_points, briefing
                        )
                        if not is_valid:
                            yield (
                                create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                                0,
                                f"Error: {error_msg}",
                                "",
                                f'<div style="color: #dc3545;">{error_msg}</div>',
                                None,
                                None,
                                None,
                                gr.update(visible=False),
                                gr.update(value="Generate Podcast", interactive=True),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                            )
                            return

                        voice_valid, voice_msg, voice_output = validate_selections(
                            voice_selections
                        )
                        if not voice_valid:
                            yield (
                                create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                                0,
                                f"Error: {voice_msg}",
                                "",
                                f'<div style="color: #dc3545;">{voice_msg}</div>',
                                None,
                                None,
                                None,
                                gr.update(visible=False),
                                gr.update(value="Generate Podcast", interactive=True),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                            )
                            return

                        print(f"[LANG] UI selected: {language}")
                        content_input = {
                            "topic": topic,
                            "key_points": key_points,
                            "briefing": briefing,
                            "num_segments": int(num_segments),
                            "language": language,
                        }

                        q: queue.Queue[_ProgressEvent | _DoneEvent | _ErrorEvent] = (
                            queue.Queue(maxsize=500)
                        )
                        cancel_event = threading.Event()

                        current_step = GenerationStep.OUTLINE
                        step_progress = 0.0
                        status_text = "Starting..."
                        last_emit_time = 0.0

                        generation_started = time.monotonic()
                        clip_durations: list[float] = []
                        current_clip_started: float = 0.0
                        current_clip_index: int = 0
                        total_clips: int = 0
                        eta_lock = threading.Lock()

                        def progress_callback(
                            step_name: str, detail: dict[str, Any] | None
                        ):
                            nonlocal \
                                last_emit_time, \
                                current_clip_started, \
                                current_clip_index, \
                                total_clips
                            if cancel_event.is_set():
                                return

                            if detail is None:
                                detail = {}

                            status = detail.get("status", "")
                            step = GenerationStep.OUTLINE

                            if step_name == "generate_clips":
                                with eta_lock:
                                    total_clips = detail.get("total", total_clips)
                                    clip_idx = detail.get("current", 0)
                                    if status == "clip_started":
                                        current_clip_started = time.monotonic()
                                        current_clip_index = clip_idx
                                    elif (
                                        status == "progress"
                                        and current_clip_started > 0
                                    ):
                                        clip_duration = (
                                            time.monotonic() - current_clip_started
                                        )
                                        if clip_duration > 0:
                                            clip_durations.append(clip_duration)
                                        current_clip_started = 0.0
                            progress = 0.0
                            status_msg = ""

                            if step_name == "generate_outline":
                                step = GenerationStep.OUTLINE
                                progress = 0.5 if status == "started" else 1.0
                                status_msg = (
                                    "Creating outline..."
                                    if status == "started"
                                    else "Outline complete"
                                )
                            elif step_name == "generate_transcript":
                                step = GenerationStep.TRANSCRIPT
                                progress = 0.5 if status == "started" else 1.0
                                status_msg = (
                                    "Generating transcript..."
                                    if status == "started"
                                    else "Transcript complete"
                                )
                            elif step_name == "generate_clips":
                                step = GenerationStep.AUDIO
                                current = detail.get("current", 0)
                                total = detail.get("total", 1)
                                segment = detail.get("segment", {})
                                speaker = segment.get("speaker", "")

                                if status == "clip_started":
                                    progress = (
                                        max(0.0, (current - 1) / total)
                                        if total > 0
                                        else 0.0
                                    )
                                    status_msg = f"Working on clip {current}/{total}: {speaker}..."
                                elif status == "progress":
                                    progress = (
                                        min(1.0, max(0.0, current / total))
                                        if total > 0
                                        else 0.0
                                    )
                                    clip_status = segment.get("status", "")
                                    if clip_status == "success":
                                        status_msg = f"Completed clip {current}/{total}"
                                    elif clip_status == "error":
                                        status_msg = f"Clip {current}/{total} failed, continuing..."
                                    else:
                                        status_msg = (
                                            f"Generating audio: {current}/{total} clips"
                                        )
                                elif status == "completed":
                                    progress = 1.0
                                    status_msg = "Audio generation complete"
                                else:
                                    progress = 0.0
                                    status_msg = "Starting audio generation..."
                            elif step_name == "combine_audio":
                                step = GenerationStep.COMBINE
                                progress = 0.5 if status == "started" else 1.0
                                status_msg = (
                                    "Combining audio..."
                                    if status == "started"
                                    else "Audio combined"
                                )

                            now = time.monotonic()
                            if (now - last_emit_time) < 0.2:
                                return
                            last_emit_time = now

                            event_data = None
                            if detail.get("outline"):
                                event_data = {"outline": detail["outline"]}
                            elif detail.get("transcript"):
                                event_data = {"transcript": detail["transcript"]}

                            evt = _ProgressEvent(
                                step=step,
                                progress=progress,
                                status=status_msg,
                                detail=str(detail),
                                data=event_data,
                            )
                            try:
                                q.put_nowait(evt)
                            except queue.Full:
                                try:
                                    q.get_nowait()
                                    q.put_nowait(evt)
                                except (queue.Empty, queue.Full):
                                    pass

                        def worker():
                            try:
                                result = podcast_orchestrator.generate_podcast(
                                    content_input=content_input,
                                    voice_selections=voice_output,
                                    quality_preset=quality_preset,
                                    progress_callback=progress_callback,
                                )
                                q.put(_DoneEvent(result=result))
                            except Exception as e:
                                q.put(
                                    _ErrorEvent(
                                        error=format_user_error(e),
                                        tb=traceback.format_exc(),
                                    )
                                )

                        worker_thread = threading.Thread(target=worker, daemon=True)
                        worker_thread.start()

                        last_yield_time = time.monotonic()
                        outline_html = (
                            '<div class="empty-state">Generating outline...</div>'
                        )
                        transcript_html = (
                            '<div class="empty-state">Waiting for transcript...</div>'
                        )

                        def render_outline_html(outline_data: dict) -> str:
                            segments = outline_data.get("segments", [])
                            if not segments:
                                return '<div class="empty-state">No segments</div>'
                            html_parts = []
                            for i, seg in enumerate(segments):
                                html_parts.append(
                                    f'<div style="margin-bottom: 0.5rem;">'
                                    f"<strong>{i + 1}. {seg.get('title', 'Segment')}</strong>"
                                    f'<br><span style="color: #666;">{seg.get("description", "")}</span>'
                                    f"</div>"
                                )
                            return "".join(html_parts)

                        def render_transcript_html(transcript_data: dict) -> str:
                            dialogues = transcript_data.get("dialogues", [])
                            if not dialogues:
                                return '<div class="empty-state">No dialogues</div>'
                            html_parts = []
                            for dlg in dialogues[:20]:
                                speaker = dlg.get("speaker", "Speaker")
                                text = dlg.get("text", "")
                                html_parts.append(
                                    f'<div style="margin-bottom: 0.5rem; padding: 0.5rem; background: #f8f9fa; border-radius: 4px;">'
                                    f"<strong>{speaker}:</strong> {text}"
                                    f"</div>"
                                )
                            if len(dialogues) > 20:
                                html_parts.append(
                                    f'<div style="color: #666; text-align: center;">... and {len(dialogues) - 20} more lines</div>'
                                )
                            return "".join(html_parts)

                        yield (
                            create_step_indicator_html(GenerationStep.OUTLINE, 0.0),
                            0,
                            "Starting podcast generation...",
                            "",
                            "",
                            None,
                            outline_html,
                            transcript_html,
                            gr.update(visible=False),
                            gr.update(value="⏳ Generating...", interactive=False),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(visible=False),
                        )

                        def format_duration(seconds: float) -> str:
                            if seconds < 60:
                                return f"{int(seconds)}s"
                            elif seconds < 3600:
                                mins = int(seconds // 60)
                                secs = int(seconds % 60)
                                return f"{mins}m {secs}s"
                            else:
                                hours = int(seconds // 3600)
                                mins = int((seconds % 3600) // 60)
                                return f"{hours}h {mins}m"

                        def get_eta_string() -> str:
                            with eta_lock:
                                if len(clip_durations) < 2 or total_clips == 0:
                                    return ""
                                avg_clip_time = sum(clip_durations) / len(
                                    clip_durations
                                )
                                completed = len(clip_durations)
                                remaining = total_clips - completed
                                if remaining <= 0:
                                    return ""
                                eta_seconds = avg_clip_time * remaining
                            return f"~{format_duration(eta_seconds)} remaining"

                        try:
                            while True:
                                try:
                                    item = q.get(timeout=0.5)
                                except queue.Empty:
                                    now = time.monotonic()
                                    if (now - last_yield_time) >= 1.5:
                                        elapsed = now - generation_started
                                        elapsed_str = (
                                            f"Elapsed: {format_duration(elapsed)}"
                                        )
                                        eta_str = get_eta_string()
                                        time_info = f"{elapsed_str}  {eta_str}".strip()

                                        yield (
                                            create_step_indicator_html(
                                                current_step, step_progress
                                            ),
                                            calculate_overall_progress(
                                                current_step, step_progress
                                            ),
                                            f"{status_text}",
                                            time_info,
                                            "",
                                            None,
                                            outline_html,
                                            transcript_html,
                                            gr.update(visible=False),
                                            gr.update(
                                                value="⏳ Generating...",
                                                interactive=False,
                                            ),
                                            gr.update(),
                                            gr.update(),
                                            gr.update(),
                                            gr.update(),
                                        )
                                        last_yield_time = now
                                    continue

                                if isinstance(item, _ProgressEvent):
                                    current_step = item.step
                                    step_progress = item.progress
                                    status_text = item.status

                                    if item.data:
                                        if "outline" in item.data:
                                            outline_html = render_outline_html(
                                                item.data["outline"]
                                            )
                                        if "transcript" in item.data:
                                            transcript_html = render_transcript_html(
                                                item.data["transcript"]
                                            )

                                    now = time.monotonic()
                                    elapsed = now - generation_started
                                    elapsed_str = f"Elapsed: {format_duration(elapsed)}"
                                    eta_str = get_eta_string()
                                    time_info = f"{elapsed_str}  {eta_str}".strip()

                                    yield (
                                        create_step_indicator_html(
                                            current_step, step_progress
                                        ),
                                        calculate_overall_progress(
                                            current_step, step_progress
                                        ),
                                        status_text,
                                        time_info,
                                        "",
                                        None,
                                        outline_html,
                                        transcript_html,
                                        gr.update(visible=False),
                                        gr.update(
                                            value="⏳ Generating...", interactive=False
                                        ),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                    )
                                    last_yield_time = now
                                    continue

                                if isinstance(item, _DoneEvent):
                                    result = item.result
                                    combined_audio_path = result.get(
                                        "combined_audio_path"
                                    )
                                    outline_path = result.get("outline_path")
                                    transcript_path = result.get("transcript_path")
                                    podcast_dir = result.get("podcast_dir")

                                    if (
                                        "empty-state" in outline_html
                                        and outline_path
                                        and Path(outline_path).exists()
                                    ):
                                        with open(outline_path) as f:
                                            outline_html = render_outline_html(
                                                json.load(f)
                                            )

                                    transcript_data = None
                                    editor_rows = []
                                    if (
                                        transcript_path
                                        and Path(transcript_path).exists()
                                    ):
                                        with open(transcript_path) as f:
                                            transcript_data = json.load(f)
                                        if "empty-state" in transcript_html:
                                            transcript_html = render_transcript_html(
                                                transcript_data
                                            )
                                        dialogues = transcript_data.get("dialogues", [])
                                        editor_rows = [
                                            [
                                                dlg.get("speaker", ""),
                                                dlg.get("text", ""),
                                            ]
                                            for dlg in dialogues
                                        ]

                                    session_info = {
                                        "podcast_dir": podcast_dir,
                                        "quality_preset": quality_preset,
                                        "language": language,
                                    }

                                    yield (
                                        create_step_indicator_html(
                                            GenerationStep.COMBINE, 1.0
                                        ),
                                        100,
                                        "Podcast generated successfully!",
                                        "",
                                        '<div style="color: #28a745;">Generation complete!</div>',
                                        combined_audio_path,
                                        outline_html,
                                        transcript_html,
                                        gr.update(
                                            value=combined_audio_path, visible=True
                                        )
                                        if combined_audio_path
                                        else gr.update(visible=False),
                                        gr.update(
                                            value="Generate Podcast", interactive=True
                                        ),
                                        transcript_data,
                                        session_info,
                                        gr.update(value=editor_rows),
                                        gr.update(visible=True),
                                    )
                                    return

                                if isinstance(item, _ErrorEvent):
                                    print(f"[Podcast Error] {item.error}\n{item.tb}")
                                    yield (
                                        create_step_indicator_html(
                                            current_step, step_progress
                                        ),
                                        calculate_overall_progress(
                                            current_step, step_progress
                                        ),
                                        f"Error: {item.error}",
                                        "",
                                        f'<div style="color: #dc3545;">Generation failed: {item.error}</div>',
                                        None,
                                        outline_html,
                                        transcript_html,
                                        gr.update(visible=False),
                                        gr.update(
                                            value="Generate Podcast", interactive=True
                                        ),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                        gr.update(),
                                    )
                                    return

                        finally:
                            cancel_event.set()

                    def regenerate_audio_from_edits(
                        editor_data,
                        session_state,
                        voice_selections,
                    ):
                        if not session_state:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: No session data available",
                                "",
                                '<div style="color: #dc3545;">Generate a podcast first</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        # Handle pandas DataFrame or list-of-lists from Gradio
                        import pandas as pd

                        if isinstance(editor_data, pd.DataFrame):
                            editor_data = editor_data.values.tolist()

                        if editor_data is None or len(editor_data) == 0:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: No transcript data to regenerate",
                                "",
                                '<div style="color: #dc3545;">Transcript is empty</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        podcast_dir = session_state.get("podcast_dir")
                        quality_preset = session_state.get("quality_preset", "standard")
                        language = session_state.get("language", "English")

                        if not podcast_dir or not Path(podcast_dir).exists():
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: Podcast directory not found",
                                "",
                                '<div style="color: #dc3545;">Session expired, generate a new podcast</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        voice_valid, voice_msg, voice_output = validate_selections(
                            voice_selections
                        )
                        if not voice_valid:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                f"Error: {voice_msg}",
                                "",
                                f'<div style="color: #dc3545;">{voice_msg}</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        dialogues = [
                            {"speaker": row[0], "text": row[1]}
                            for row in editor_data
                            if len(row) >= 2 and row[0] and row[1]
                        ]

                        if not dialogues:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                "Error: No valid dialogue entries",
                                "",
                                '<div style="color: #dc3545;">Add speaker and text to dialogues</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )
                            return

                        yield (
                            create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                            30,
                            "Regenerating audio from edited transcript...",
                            "",
                            "",
                            None,
                            gr.update(visible=False),
                            gr.update(value="⏳ Regenerating...", interactive=False),
                        )

                        try:
                            transcript = podcast_orchestrator.transcript_from_struct(
                                dialogues
                            )
                            speaker_profile = create_speaker_profile(voice_output)

                            transcript_path = Path(podcast_dir) / "transcript.json"
                            with open(transcript_path, "w") as f:
                                json.dump(transcript.model_dump(), f, indent=2)

                            q: queue.Queue = queue.Queue(maxsize=100)
                            result_holder: list = []
                            error_holder: list = []

                            def progress_cb(step: str, detail: dict | None):
                                if detail:
                                    q.put({"step": step, "detail": detail})

                            def worker():
                                try:
                                    clips, combined = (
                                        podcast_orchestrator.generate_audio_only(
                                            transcript=transcript,
                                            speaker_profile=speaker_profile,
                                            podcast_dir=Path(podcast_dir),
                                            quality_preset=quality_preset,
                                            language=language,
                                            progress_callback=progress_cb,
                                        )
                                    )
                                    result_holder.append(str(combined))
                                except Exception as e:
                                    error_holder.append(str(e))

                            worker_thread = threading.Thread(target=worker, daemon=True)
                            worker_thread.start()

                            while worker_thread.is_alive():
                                try:
                                    item = q.get(timeout=0.5)
                                    detail = item.get("detail", {})
                                    status = detail.get("status", "")
                                    current = detail.get("current", 0)
                                    total = detail.get("total", 1)

                                    if item.get("step") == "generate_clips":
                                        progress = int(
                                            30 + (current / max(total, 1)) * 50
                                        )
                                        yield (
                                            create_step_indicator_html(
                                                GenerationStep.AUDIO,
                                                current / max(total, 1),
                                            ),
                                            progress,
                                            f"Generating clip {current}/{total}...",
                                            "",
                                            "",
                                            None,
                                            gr.update(visible=False),
                                            gr.update(
                                                value="⏳ Regenerating...",
                                                interactive=False,
                                            ),
                                        )
                                    elif item.get("step") == "combine_audio":
                                        yield (
                                            create_step_indicator_html(
                                                GenerationStep.COMBINE, 0.5
                                            ),
                                            85,
                                            "Combining audio clips...",
                                            "",
                                            "",
                                            None,
                                            gr.update(visible=False),
                                            gr.update(
                                                value="⏳ Regenerating...",
                                                interactive=False,
                                            ),
                                        )
                                except queue.Empty:
                                    continue

                            worker_thread.join()

                            if error_holder:
                                yield (
                                    create_step_indicator_html(
                                        GenerationStep.AUDIO, 0.0
                                    ),
                                    0,
                                    f"Error: {error_holder[0]}",
                                    "",
                                    f'<div style="color: #dc3545;">Regeneration failed: {error_holder[0]}</div>',
                                    None,
                                    gr.update(visible=False),
                                    gr.update(
                                        value="Regenerate Audio from Edits",
                                        interactive=True,
                                    ),
                                )
                                return

                            if result_holder:
                                combined_path = result_holder[0]
                                yield (
                                    create_step_indicator_html(
                                        GenerationStep.COMBINE, 1.0
                                    ),
                                    100,
                                    "Audio regenerated successfully!",
                                    "",
                                    '<div style="color: #28a745;">Regeneration complete!</div>',
                                    combined_path,
                                    gr.update(value=combined_path, visible=True),
                                    gr.update(
                                        value="Regenerate Audio from Edits",
                                        interactive=True,
                                    ),
                                )
                            else:
                                yield (
                                    create_step_indicator_html(
                                        GenerationStep.AUDIO, 0.0
                                    ),
                                    0,
                                    "Error: No audio generated",
                                    "",
                                    '<div style="color: #dc3545;">Unknown error occurred</div>',
                                    None,
                                    gr.update(visible=False),
                                    gr.update(
                                        value="Regenerate Audio from Edits",
                                        interactive=True,
                                    ),
                                )

                        except Exception as e:
                            yield (
                                create_step_indicator_html(GenerationStep.AUDIO, 0.0),
                                0,
                                f"Error: {str(e)}",
                                "",
                                f'<div style="color: #dc3545;">Regeneration failed: {str(e)}</div>',
                                None,
                                gr.update(visible=False),
                                gr.update(
                                    value="Regenerate Audio from Edits",
                                    interactive=True,
                                ),
                            )

                    podcast_topic.change(
                        fn=update_topic_char_count,
                        inputs=[podcast_topic],
                        outputs=[podcast_topic_chars],
                    )

                    all_slot_inputs = []
                    for slot_role, slot_voice in podcast_speaker_slots:
                        all_slot_inputs.extend([slot_role, slot_voice])

                    for slot_role, slot_voice in podcast_speaker_slots:
                        slot_role.change(
                            fn=build_voice_selections_from_slots,
                            inputs=all_slot_inputs,
                            outputs=[
                                podcast_voice_selections_state,
                                podcast_voice_summary,
                            ],
                        )
                        slot_voice.change(
                            fn=build_voice_selections_from_slots,
                            inputs=all_slot_inputs,
                            outputs=[
                                podcast_voice_selections_state,
                                podcast_voice_summary,
                            ],
                        )

                    podcast_generate_btn.click(
                        fn=run_podcast_generation,
                        inputs=[
                            podcast_topic,
                            podcast_key_points,
                            podcast_briefing,
                            podcast_num_segments,
                            podcast_voice_selections_state,
                            podcast_quality_preset,
                            podcast_language,
                        ],
                        outputs=[
                            podcast_step_indicator,
                            podcast_overall_progress,
                            podcast_status,
                            podcast_time_remaining,
                            podcast_error_display,
                            podcast_final_audio,
                            podcast_outline_html,
                            podcast_transcript_html,
                            podcast_download,
                            podcast_generate_btn,
                            podcast_transcript_state,
                            podcast_session_state,
                            podcast_transcript_editor,
                            podcast_edit_accordion,
                        ],
                        concurrency_limit=1,
                        concurrency_id="podcast_generation",
                    )

                    podcast_regenerate_btn.click(
                        fn=regenerate_audio_from_edits,
                        inputs=[
                            podcast_transcript_editor,
                            podcast_session_state,
                            podcast_voice_selections_state,
                        ],
                        outputs=[
                            podcast_step_indicator,
                            podcast_overall_progress,
                            podcast_status,
                            podcast_time_remaining,
                            podcast_error_display,
                            podcast_final_audio,
                            podcast_download,
                            podcast_regenerate_btn,
                        ],
                        concurrency_limit=1,
                        concurrency_id="podcast_regeneration",
                    )

                    for i, preview_btn in enumerate(podcast_preview_buttons):
                        preview_btn.click(
                            fn=play_podcast_preview,
                            inputs=[podcast_speaker_slots[i][1]],
                            outputs=[podcast_preview_audio],
                        )

                    podcast_history_dropdown.change(
                        fn=load_podcast_history_item,
                        inputs=[podcast_history_dropdown],
                        outputs=[podcast_history_audio, podcast_history_metadata],
                    ).then(
                        fn=lambda: False,
                        outputs=[podcast_history_delete_confirm],
                    )

                    podcast_history_refresh.click(
                        fn=lambda: gr.update(choices=get_podcast_history_choices()),
                        inputs=[],
                        outputs=[podcast_history_dropdown],
                    )

                    podcast_history_delete.click(
                        fn=delete_podcast_history_item,
                        inputs=[
                            podcast_history_dropdown,
                            podcast_history_delete_confirm,
                        ],
                        outputs=[
                            podcast_history_metadata,
                            podcast_history_dropdown,
                            podcast_history_audio,
                            podcast_history_delete_confirm,
                        ],
                    )
                    podcast_history_favorite.click(
                        fn=toggle_favorite,
                        inputs=[podcast_history_dropdown],
                        outputs=[
                            podcast_history_metadata,
                            podcast_history_display,
                            podcast_history_dropdown,
                        ],
                    )
                    podcast_history_search.change(
                        fn=search_history,
                        inputs=[podcast_history_search, podcast_history_favorites],
                        outputs=[podcast_history_display],
                        show_progress="hidden",
                    )
                    podcast_history_favorites.change(
                        fn=search_history,
                        inputs=[podcast_history_search, podcast_history_favorites],
                        outputs=[podcast_history_display],
                        show_progress="hidden",
                    )

        with gr.Column(scale=1, elem_classes=["compact-params-panel"]):
            gr.HTML('<div class="panel-header-compact">Parameters</div>')

            save_indicator = gr.HTML(
                value='<span class="save-indicator">Settings saved</span>'
            )

            gr.HTML('<div class="preset-section">')
            gr.HTML(
                '<div style="font-size:0.75rem;font-weight:600;color:var(--gray-700);margin-bottom:0.5rem;">Quick Presets</div>'
            )
            with gr.Row(elem_classes=["preset-btn-group"]):
                preset_fast = gr.Button(
                    "Fast", size="sm", elem_classes=["preset-btn-lg"]
                )
                preset_balanced = gr.Button(
                    "Balanced", size="sm", elem_classes=["preset-btn-lg"]
                )
                preset_quality = gr.Button(
                    "Quality", size="sm", elem_classes=["preset-btn-lg"]
                )
            reset_btn = gr.Button("Reset", size="sm", variant="secondary")
            gr.HTML("</div>")

            with gr.Accordion("Basic Parameters", open=True):
                param_temp = gr.Slider(
                    0.1,
                    1.5,
                    value=min(settings["temperature"], 1.5),
                    step=0.05,
                    label="Temperature",
                    info=PARAM_TOOLTIPS["temperature"],
                )

                with gr.Row(elem_classes=["compact-slider-row"]):
                    param_top_k = gr.Slider(
                        1,
                        100,
                        value=settings["top_k"],
                        step=1,
                        label="Top-K",
                        info=PARAM_TOOLTIPS["top_k"],
                    )
                    param_top_p = gr.Slider(
                        0.1,
                        1.0,
                        value=settings["top_p"],
                        step=0.05,
                        label="Top-P",
                        info=PARAM_TOOLTIPS["top_p"],
                    )

            with gr.Accordion("Advanced Parameters", open=False):
                with gr.Row(elem_classes=["compact-slider-row"]):
                    param_rep_pen = gr.Slider(
                        1.0,
                        2.0,
                        value=settings["repetition_penalty"],
                        step=0.01,
                        label="Repetition Penalty",
                        info=PARAM_TOOLTIPS["repetition_penalty"],
                    )
                    param_max_tokens = gr.Slider(
                        512,
                        8192,
                        value=settings["max_new_tokens"],
                        step=256,
                        label="Max Tokens",
                        info=PARAM_TOOLTIPS["max_new_tokens"],
                    )

                gr.HTML(
                    '<div style="font-size:0.7rem;color:var(--gray-500);margin:0.5rem 0 0.25rem;font-weight:500;">Subtalker Model (defaults recommended)</div>'
                )

                with gr.Row(elem_classes=["compact-slider-row"]):
                    param_sub_temp = gr.Slider(
                        0.1,
                        1.5,
                        value=min(settings["subtalker_temperature"], 1.5),
                        step=0.05,
                        label="Sub Temperature",
                        info=PARAM_TOOLTIPS["subtalker_temperature"],
                    )
                    param_sub_top_k = gr.Slider(
                        1,
                        100,
                        value=settings["subtalker_top_k"],
                        step=1,
                        label="Sub Top-K",
                        info=PARAM_TOOLTIPS["subtalker_top_k"],
                    )

                param_sub_top_p = gr.Slider(
                    0.1,
                    1.0,
                    value=settings["subtalker_top_p"],
                    step=0.05,
                    label="Sub Top-P",
                    info=PARAM_TOOLTIPS["subtalker_top_p"],
                )

    all_param_sliders = [
        param_temp,
        param_top_k,
        param_top_p,
        param_rep_pen,
        param_max_tokens,
        param_sub_temp,
        param_sub_top_k,
        param_sub_top_p,
    ]

    cv_text.change(fn=update_char_count, inputs=[cv_text], outputs=[cv_char_count])
    vc_test_text.change(
        fn=update_char_count, inputs=[vc_test_text], outputs=[vc_test_char_count]
    )
    sv_text.change(fn=update_char_count, inputs=[sv_text], outputs=[sv_char_count])

    for slider in all_param_sliders:
        slider.change(
            fn=on_param_change, inputs=all_param_sliders, outputs=[save_indicator]
        )

    preset_fast.click(
        fn=lambda: apply_preset("fast"), outputs=all_param_sliders + [save_indicator]
    )
    preset_balanced.click(
        fn=lambda: apply_preset("balanced"),
        outputs=all_param_sliders + [save_indicator],
    )
    preset_quality.click(
        fn=lambda: apply_preset("quality"), outputs=all_param_sliders + [save_indicator]
    )

    reset_btn.click(fn=reset_params, outputs=all_param_sliders + [save_indicator])

    podcast_quality_preset.change(
        fn=lambda preset: apply_podcast_preset(preset)[:-1],
        inputs=[podcast_quality_preset],
        outputs=all_param_sliders + [podcast_num_segments],
    )

    cv_history_dropdown.change(
        fn=play_history_item_with_details,
        inputs=[cv_history_dropdown],
        outputs=[cv_history_audio, cv_history_text, cv_history_params],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    ).then(
        fn=lambda: False,
        outputs=[cv_history_delete_confirm],
    )
    cv_history_refresh.click(
        fn=lambda: gr.update(choices=get_history_choices()),
        outputs=[cv_history_dropdown],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    cv_history_apply.click(
        fn=apply_history_params,
        inputs=[cv_history_dropdown],
        outputs=all_param_sliders + [save_indicator],
        concurrency_id="history",
        concurrency_limit=None,
    )
    cv_history_delete.click(
        fn=delete_history_item,
        inputs=[cv_history_dropdown, cv_history_delete_confirm],
        outputs=[
            cv_status,
            cv_history_dropdown,
            cv_history_audio,
            cv_history_delete_confirm,
        ],
        concurrency_id="history",
        concurrency_limit=None,
    )
    cv_history_favorite.click(
        fn=toggle_favorite,
        inputs=[cv_history_dropdown],
        outputs=[cv_status, cv_history_display, cv_history_dropdown],
        concurrency_id="history",
        concurrency_limit=None,
    )
    cv_history_search.change(
        fn=search_history,
        inputs=[cv_history_search, cv_history_favorites],
        outputs=[cv_history_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    cv_history_favorites.change(
        fn=search_history,
        inputs=[cv_history_search, cv_history_favorites],
        outputs=[cv_history_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )

    sv_history_dropdown.change(
        fn=play_history_item_with_details,
        inputs=[sv_history_dropdown],
        outputs=[sv_history_audio, sv_history_text, sv_history_params],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    ).then(
        fn=lambda: False,
        outputs=[sv_history_delete_confirm],
    )
    sv_history_refresh.click(
        fn=lambda: gr.update(choices=get_history_choices()),
        outputs=[sv_history_dropdown],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    sv_history_apply.click(
        fn=apply_history_params,
        inputs=[sv_history_dropdown],
        outputs=all_param_sliders + [save_indicator],
        concurrency_id="history",
        concurrency_limit=None,
    )
    sv_history_delete.click(
        fn=delete_history_item,
        inputs=[sv_history_dropdown, sv_history_delete_confirm],
        outputs=[
            sv_status,
            sv_history_dropdown,
            sv_history_audio,
            sv_history_delete_confirm,
        ],
        concurrency_id="history",
        concurrency_limit=None,
    )
    sv_history_favorite.click(
        fn=toggle_favorite,
        inputs=[sv_history_dropdown],
        outputs=[sv_status, sv_history_display, sv_history_dropdown],
        concurrency_id="history",
        concurrency_limit=None,
    )
    sv_history_search.change(
        fn=search_history,
        inputs=[sv_history_search, sv_history_favorites],
        outputs=[sv_history_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )
    sv_history_favorites.change(
        fn=search_history,
        inputs=[sv_history_search, sv_history_favorites],
        outputs=[sv_history_display],
        concurrency_id="history",
        concurrency_limit=None,
        show_progress="hidden",
    )

    cv_btn.click(
        fn=generate_custom_voice,
        inputs=[cv_text, cv_model, cv_speaker, cv_language, cv_instruct]
        + all_param_sliders,
        outputs=[cv_audio, cv_status],
        concurrency_limit=1,
        concurrency_id="generation",
    ).then(
        fn=lambda: gr.update(choices=get_history_choices()),
        outputs=[cv_history_dropdown],
        show_progress="hidden",
    )

    vc_clone_btn.click(
        fn=clone_voice,
        inputs=[vc_ref_audio, vc_ref_text, vc_model, vc_test_text, vc_language]
        + all_param_sliders,
        outputs=[vc_output, current_prompt_data, current_clone_model, vc_status],
        concurrency_limit=1,
        concurrency_id="generation",
    )

    vc_auto_transcribe_btn.click(
        fn=auto_transcribe_audio,
        inputs=[vc_ref_audio],
        outputs=[vc_ref_text],
    )

    vc_save_btn.click(
        fn=save_cloned_voice,
        inputs=[
            vc_name,
            vc_description,
            vc_style_note,
            vc_ref_audio,
            vc_ref_text,
            current_prompt_data,
            current_clone_model,
        ],
        outputs=[vc_save_status, sv_voice_dropdown],
    ).then(
        fn=lambda: [gr.update(choices=_get_podcast_voice_choices()) for _ in range(4)],
        outputs=[slot[1] for slot in podcast_speaker_slots],
    ).then(
        fn=lambda: gr.update(choices=_get_persona_voice_choices(), value=None),
        outputs=[persona_voice_dropdown],
    )

    sv_refresh_btn.click(
        fn=lambda: gr.update(choices=get_saved_voice_choices()),
        outputs=[sv_voice_dropdown],
    )

    sv_voice_dropdown.change(
        fn=get_voice_details,
        inputs=[sv_voice_dropdown],
        outputs=[
            sv_description,
            sv_style_note,
            sv_ref_text,
            sv_model_info,
            sv_ref_audio,
        ],
    ).then(fn=lambda: False, outputs=[sv_delete_confirm])

    sv_delete_btn.click(
        fn=delete_saved_voice,
        inputs=[sv_voice_dropdown, sv_delete_confirm],
        outputs=[sv_delete_status, sv_voice_dropdown, sv_ref_audio, sv_delete_confirm],
    )

    sv_generate_btn.click(
        fn=generate_with_saved_voice,
        inputs=[sv_text, sv_voice_dropdown, sv_language] + all_param_sliders,
        outputs=[sv_audio, sv_status],
        concurrency_limit=1,
        concurrency_id="generation",
    ).then(
        fn=lambda: gr.update(choices=get_history_choices()),
        outputs=[sv_history_dropdown],
        show_progress="hidden",
    )

    def refresh_podcast_voice_dropdowns():
        choices = _get_podcast_voice_choices()
        return [gr.update(choices=choices) for _ in range(4)]

    podcast_refresh_voices_btn.click(
        fn=refresh_podcast_voice_dropdowns,
        outputs=[slot[1] for slot in podcast_speaker_slots],
    )

if __name__ == "__main__":
    print("Starting Qwen3-TTS Studio...")
    print("=" * 50)
    print("Features:")
    print("  • Visible parameters with auto-save")
    print("  • Quick presets: Fast, Balanced, Quality")
    print("  • Character count with warnings")
    print("  • Generation time tracking")
    print("  • History with search & favorites")
    print("  • Export history to ZIP")
    print("=" * 50)
    demo.queue(default_concurrency_limit=1)
    demo.launch(server_name="127.0.0.1", server_port=7860, css=custom_css)
