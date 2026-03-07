"""Podcast generation orchestrator."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, cast

from podcast import outline as outline_generator
from podcast import transcript as transcript_generator
from podcast.llm_client import LLMConfig
from audio import batch as batch_processor
from audio import combiner as audio_combiner
from storage import history as storage
from storage import voice as voice_selection
from podcast.models import Outline, Segment, Speaker, SpeakerProfile, Transcript
from storage.persona_models import Persona
from storage.persona import load_persona

ProgressCallback = Callable[[str, dict[str, object] | None], None]


def _timestamped_podcast_name() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"podcast_{timestamp}"


def _parse_key_points(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if not isinstance(raw, str):
        return [str(raw).strip()]
    points: list[str] = []
    for line in raw.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith(("- ", "* ")):
            cleaned = cleaned[2:].strip()
        if cleaned:
            points.append(cleaned)
    if not points and raw.strip():
        points = [raw.strip()]
    return points


LANGUAGE_CODE_MAP = {
    "english": "en", "korean": "ko", "japanese": "ja", "chinese": "zh",
    "spanish": "es", "french": "fr", "german": "de", "italian": "it",
    "portuguese": "pt", "russian": "ru",
}

def _normalize_language_code(lang: str) -> str:
    lang_lower = lang.lower().strip()
    if lang_lower in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[lang_lower]
    if lang_lower in LANGUAGE_CODE_MAP.values():
        return lang_lower
    return "en"


def _resolve_tts_params(quality_preset: object, language: str = "en") -> dict[str, object]:
    language = _normalize_language_code(language)
    defaults: dict[str, object] = {
        "model_name": "1.7B-CustomVoice",
        "temperature": 0.3,
        "top_k": 50,
        "top_p": 0.85,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "subtalker_temperature": 0.3,
        "subtalker_top_k": 50,
        "subtalker_top_p": 0.85,
        "language": language,
        "instruct": None,
    }
    presets = {
         # UI names
         "quick": {"temperature": 0.5, "top_p": 0.9, "max_new_tokens": 768},
         "standard": {},
         "premium": {"temperature": 0.2, "top_p": 0.8, "max_new_tokens": 1400},
         # Legacy names (backwards compatibility)
         "draft": {"temperature": 0.5, "top_p": 0.9, "max_new_tokens": 768},
         "high": {"temperature": 0.2, "top_p": 0.8, "max_new_tokens": 1400},
     }
    if isinstance(quality_preset, dict):
        return {**defaults, **quality_preset}
    if isinstance(quality_preset, str):
        preset = presets.get(quality_preset.strip().lower(), {})
        return {**defaults, **preset}
    return defaults


def _notify(
    callback: ProgressCallback | None,
    step: str,
    detail: dict[str, object] | None,
) -> None:
    if callback is not None:
        callback(step, detail)


def _load_personas_for_speakers(speaker_profile: SpeakerProfile) -> dict[str, Persona]:
    """Load personas for all speakers in profile."""
    personas = {}
    for speaker in speaker_profile.speakers:
        persona = load_persona(speaker.voice_id, speaker.type)
        if persona:
            personas[speaker.voice_id] = persona
    return personas


def generate_podcast(
    content_input: dict[str, object],
    voice_selections: list[dict[str, str]],
    quality_preset: str | dict[str, object] | None,
    progress_callback: ProgressCallback | None = None,
    llm_config: LLMConfig | None = None,
) -> dict[str, str]:
    """
    Orchestrate full podcast generation workflow.

    Args:
        content_input: Dict with keys like topic, key_points, briefing, num_segments.
        voice_selections: List of voice selection dicts.
        quality_preset: Name of preset or TTS params dict.
        progress_callback: Optional callback(step_name, detail) invoked at each step.

    Returns:
        Dict with paths to artifacts generated during the workflow.
    """
    topic = str(content_input.get("topic", "")).strip()
    if not topic:
        raise ValueError("content_input must include a non-empty 'topic'.")

    key_points = _parse_key_points(content_input.get("key_points", []))
    briefing = str(content_input.get("briefing", "")).strip()
    language = str(content_input.get("language", "English")).strip()
    num_segments_raw = content_input.get("num_segments", 5)
    if isinstance(num_segments_raw, bool) or not isinstance(num_segments_raw, (int, str)):
        raise ValueError("num_segments must be an integer.")
    try:
        num_segments = int(num_segments_raw)
    except ValueError as exc:
        raise ValueError("num_segments must be an integer.") from exc

    tts_params = _resolve_tts_params(quality_preset, language)
    started_at = datetime.now(timezone.utc)

    podcast_dir: Path | None = None
    try:
        _notify(progress_callback, "create_directory", {"status": "started"})
        podcast_name = _timestamped_podcast_name()
        podcast_dir = storage.create_podcast_directory(podcast_name)
        clips_dir = podcast_dir / "clips"
        _notify(
            progress_callback,
            "create_directory",
            {"status": "completed", "podcast_dir": str(podcast_dir)},
        )

        _notify(progress_callback, "generate_outline", {"status": "started"})
        speaker_profile = voice_selection.create_speaker_profile(voice_selections)
        personas = _load_personas_for_speakers(speaker_profile)
        outline = cast(
            Outline,
            outline_generator.generate_outline(
                topic=topic,
                key_points=key_points,
                briefing=briefing,
                num_segments=num_segments,
                speakers=speaker_profile.speakers,
                personas=personas,
                llm_config=llm_config,
            ),
        )
        _notify(progress_callback, "generate_outline", {"status": "completed", "outline": outline.model_dump()})

        _notify(progress_callback, "generate_transcript", {"status": "started"})
        transcript = cast(
            Transcript,
            transcript_generator.generate_transcript(
                outline=outline,
                topic=topic,
                briefing=briefing,
                speakers=speaker_profile.speakers,
                personas=personas,
                language=language,
                llm_config=llm_config,
            ),
        )
        _notify(progress_callback, "generate_transcript", {"status": "completed", "transcript": transcript.model_dump()})

        _notify(progress_callback, "save_artifacts", {"status": "started"})
        outline_path = storage.save_outline(outline, podcast_dir)
        transcript_path = storage.save_transcript(transcript, podcast_dir)
        _notify(
            progress_callback,
            "save_artifacts",
            {
                "status": "completed",
                "outline_path": str(outline_path),
                "transcript_path": str(transcript_path),
            },
        )

        _notify(progress_callback, "generate_clips", {"status": "started"})

        def clip_progress(
            current: int,
            total: int,
            segment_info: dict[str, object],
        ) -> None:
            clip_status = "clip_started" if segment_info.get("status") == "started" else "progress"
            _notify(
                progress_callback,
                "generate_clips",
                {
                    "status": clip_status,
                    "current": current,
                    "total": total,
                    "segment": segment_info,
                },
            )

        clip_paths = batch_processor.generate_all_clips(
            transcript=transcript,
            speaker_profile=speaker_profile,
            params=cast(dict[str, object], tts_params),
            clips_dir=clips_dir,
            progress_callback=clip_progress,
        )
        _notify(
            progress_callback,
            "generate_clips",
            {"status": "completed", "clip_count": len(clip_paths)},
        )

        _notify(progress_callback, "combine_audio", {"status": "started"})
        combined_audio_path = audio_combiner.combine_audio_clips(
            clips_dir=clips_dir,
            output_path=podcast_dir / "final_podcast.mp3",
        )
        _notify(
            progress_callback,
            "combine_audio",
            {"status": "completed", "output_path": str(combined_audio_path)},
        )

        _notify(progress_callback, "save_metadata", {"status": "started"})
        finished_at = datetime.now(timezone.utc)
        metadata = {
             "topic": topic,
             "briefing": briefing,
             "key_points": key_points,
             "num_segments": num_segments,
             "language": language,
             "speakers": [speaker.model_dump() for speaker in speaker_profile.speakers],
             "tts_params": tts_params,
             "quality_preset": quality_preset,
             "llm_provider": llm_config.provider.value if llm_config else "openai",
             "llm_model": llm_config.model if llm_config else "default",
             "created_at": started_at.isoformat(),
             "completed_at": finished_at.isoformat(),
         }
        metadata_path = podcast_dir / "metadata.json"
        _ = metadata_path.write_text(json.dumps(metadata, indent=2))
        _notify(
            progress_callback,
            "save_metadata",
            {"status": "completed", "metadata_path": str(metadata_path)},
        )

        return {
            "podcast_dir": str(podcast_dir),
            "outline_path": str(outline_path),
            "transcript_path": str(transcript_path),
            "clips_dir": str(clips_dir),
            "combined_audio_path": str(combined_audio_path),
            "metadata_path": str(metadata_path),
        }
    except Exception as exc:
        _notify(
            progress_callback,
            "error",
            {"status": "failed", "error": str(exc)},
        )
        if podcast_dir is not None and podcast_dir.exists():
            shutil.rmtree(podcast_dir, ignore_errors=True)
        raise


def generate_outline_only(
    topic: str,
    key_points: list[str] | str,
    briefing: str,
    num_segments: int,
    voice_selections: list[dict[str, str]],
    progress_callback: ProgressCallback | None = None,
    llm_config: LLMConfig | None = None,
) -> tuple[Outline, SpeakerProfile]:
    if not topic.strip():
        raise ValueError("Topic cannot be empty.")
    
    if isinstance(key_points, str):
        key_points = _parse_key_points(key_points)
    
    _notify(progress_callback, "generate_outline", {"status": "started"})
    
    speaker_profile = voice_selection.create_speaker_profile(voice_selections)
    personas = _load_personas_for_speakers(speaker_profile)
    
    outline = cast(
        Outline,
        outline_generator.generate_outline(
            topic=topic,
            key_points=key_points,
            briefing=briefing,
            num_segments=num_segments,
            speakers=speaker_profile.speakers,
            personas=personas,
            llm_config=llm_config,
        ),
    )

    _notify(progress_callback, "generate_outline", {"status": "completed", "outline": outline.model_dump()})

    return outline, speaker_profile


def generate_transcript_only(
    outline: Outline,
    topic: str,
    briefing: str,
    speaker_profile: SpeakerProfile,
    language: str = "English",
    progress_callback: ProgressCallback | None = None,
    llm_config: LLMConfig | None = None,
) -> Transcript:
    _notify(progress_callback, "generate_transcript", {"status": "started"})
    
    personas = _load_personas_for_speakers(speaker_profile)
    
    transcript = cast(
        Transcript,
        transcript_generator.generate_transcript(
            outline=outline,
            topic=topic,
            briefing=briefing,
            speakers=speaker_profile.speakers,
            personas=personas,
            language=language,
            llm_config=llm_config,
        ),
    )

    _notify(progress_callback, "generate_transcript", {"status": "completed", "transcript": transcript.model_dump()})

    return transcript


def generate_audio_only(
    transcript: Transcript,
    speaker_profile: SpeakerProfile,
    podcast_dir: Path,
    quality_preset: str | dict[str, object] | None = "standard",
    language: str = "en",
    progress_callback: ProgressCallback | None = None,
) -> tuple[list[Path], Path]:
    tts_params = _resolve_tts_params(quality_preset, language)
    clips_dir = podcast_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    _notify(progress_callback, "generate_clips", {"status": "started"})
    
    def clip_progress(current: int, total: int, segment_info: dict[str, object]) -> None:
        _notify(
            progress_callback,
            "generate_clips",
            {"status": "progress", "current": current, "total": total, "segment": segment_info},
        )
    
    clip_paths = batch_processor.generate_all_clips(
        transcript=transcript,
        speaker_profile=speaker_profile,
        params=cast(dict[str, object], tts_params),
        clips_dir=clips_dir,
        progress_callback=clip_progress,
    )
    
    _notify(progress_callback, "generate_clips", {"status": "completed", "clip_count": len(clip_paths)})
    
    _notify(progress_callback, "combine_audio", {"status": "started"})
    
    combined_audio_path = audio_combiner.combine_audio_clips(
        clips_dir=clips_dir,
        output_path=podcast_dir / "final_podcast.mp3",
    )
    
    _notify(progress_callback, "combine_audio", {"status": "completed", "output_path": str(combined_audio_path)})
    
    return [Path(p) for p in clip_paths], Path(combined_audio_path)


def create_podcast_directory() -> Path:
    podcast_name = _timestamped_podcast_name()
    return storage.create_podcast_directory(podcast_name)


def save_outline_to_dir(outline: Outline, podcast_dir: Path) -> Path:
    return storage.save_outline(outline, podcast_dir)


def save_transcript_to_dir(transcript: Transcript, podcast_dir: Path) -> Path:
    return storage.save_transcript(transcript, podcast_dir)


def outline_from_struct(segments: list[dict[str, str]]) -> Outline:
    return Outline(
        segments=[
            Segment(
                title=seg.get("title", "Untitled"),
                description=seg.get("description", ""),
                size=seg.get("size", "medium"),
            )
            for seg in segments
        ]
    )


def transcript_from_struct(dialogues: list[dict[str, str]]) -> Transcript:
    from podcast.models import Dialogue
    return Transcript(
        dialogues=[
            Dialogue(
                speaker=dlg.get("speaker", "narrator"),
                text=dlg.get("text", ""),
            )
            for dlg in dialogues
        ]
    )


def generate_podcast_from_script(
    dialogues: list[dict[str, str]],
    voice_selections: list[dict[str, str]],
    quality_preset: str | dict[str, object] | None = "standard",
    language: str = "English",
    title: str = "",
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    """
    Generate a podcast from a user-provided script (no LLM involved).

    This is the orchestrator entry-point for the "Custom Script" feature.
    It mirrors the artifact/metadata structure of ``generate_podcast`` so that
    the result integrates with history, progress UI, and audio output.

    Args:
        dialogues: Pre-parsed dialogue list from script_parser.
        voice_selections: List of voice selection dicts.
        quality_preset: TTS quality preset name or params dict.
        language: Language name or code.
        title: Optional episode title for metadata (derived from script if empty).
        progress_callback: Optional progress callback.

    Returns:
        Dict with paths to artifacts generated during the workflow.
    """
    if not dialogues:
        raise ValueError("No dialogue lines provided.")

    tts_params = _resolve_tts_params(quality_preset, language)
    started_at = datetime.now(timezone.utc)

    podcast_dir: Path | None = None
    try:
        _notify(progress_callback, "create_directory", {"status": "started"})
        podcast_name = _timestamped_podcast_name()
        podcast_dir = storage.create_podcast_directory(podcast_name)
        _notify(
            progress_callback,
            "create_directory",
            {"status": "completed", "podcast_dir": str(podcast_dir)},
        )

        # Build transcript from structured dialogues.
        _notify(progress_callback, "generate_transcript", {"status": "started"})
        transcript = transcript_from_struct(dialogues)
        _notify(
            progress_callback,
            "generate_transcript",
            {"status": "completed", "transcript": transcript.model_dump()},
        )

        # Save transcript artifact.
        _notify(progress_callback, "save_artifacts", {"status": "started"})
        transcript_path = storage.save_transcript(transcript, podcast_dir)
        _notify(
            progress_callback,
            "save_artifacts",
            {"status": "completed", "transcript_path": str(transcript_path)},
        )

        # Create speaker profile and generate audio.
        speaker_profile = voice_selection.create_speaker_profile(voice_selections)

        clips_dir = podcast_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        _notify(progress_callback, "generate_clips", {"status": "started"})

        def clip_progress(
            current: int,
            total: int,
            segment_info: dict[str, object],
        ) -> None:
            clip_status = (
                "clip_started"
                if segment_info.get("status") == "started"
                else "progress"
            )
            _notify(
                progress_callback,
                "generate_clips",
                {
                    "status": clip_status,
                    "current": current,
                    "total": total,
                    "segment": segment_info,
                },
            )

        clip_paths = batch_processor.generate_all_clips(
            transcript=transcript,
            speaker_profile=speaker_profile,
            params=cast(dict[str, object], tts_params),
            clips_dir=clips_dir,
            progress_callback=clip_progress,
        )
        _notify(
            progress_callback,
            "generate_clips",
            {"status": "completed", "clip_count": len(clip_paths)},
        )

        _notify(progress_callback, "combine_audio", {"status": "started"})
        combined_audio_path = audio_combiner.combine_audio_clips(
            clips_dir=clips_dir,
            output_path=podcast_dir / "final_podcast.mp3",
        )
        _notify(
            progress_callback,
            "combine_audio",
            {"status": "completed", "output_path": str(combined_audio_path)},
        )

        # Save metadata.
        _notify(progress_callback, "save_metadata", {"status": "started"})
        finished_at = datetime.now(timezone.utc)

        # Derive title from first dialogue if not provided.
        if not title:
            first_text = dialogues[0].get("text", "")
            title = (first_text[:60] + "...") if len(first_text) > 60 else first_text
            if not title:
                title = "Custom Script Podcast"

        speakers_in_script = list(dict.fromkeys(
            dlg.get("speaker", "") for dlg in dialogues
        ))

        metadata = {
            "topic": title,
            "mode": "custom_script",
            "language": language,
            "speakers": [s.model_dump() for s in speaker_profile.speakers],
            "speakers_in_script": speakers_in_script,
            "tts_params": tts_params,
            "quality_preset": quality_preset,
            "dialogue_count": len(dialogues),
            "created_at": started_at.isoformat(),
            "completed_at": finished_at.isoformat(),
        }
        metadata_path = podcast_dir / "metadata.json"
        _ = metadata_path.write_text(json.dumps(metadata, indent=2))
        _notify(
            progress_callback,
            "save_metadata",
            {"status": "completed", "metadata_path": str(metadata_path)},
        )

        return {
            "podcast_dir": str(podcast_dir),
            "transcript_path": str(transcript_path),
            "clips_dir": str(clips_dir),
            "combined_audio_path": str(combined_audio_path),
            "metadata_path": str(metadata_path),
        }
    except Exception as exc:
        _notify(
            progress_callback,
            "error",
            {"status": "failed", "error": str(exc)},
        )
        if podcast_dir is not None and podcast_dir.exists():
            shutil.rmtree(podcast_dir, ignore_errors=True)
        raise

if __name__ == "__main__":
    def mock_progress(step: str, detail: dict[str, object] | None) -> None:
        print(f"[{step}] {detail}")

    def mock_create_profile(voice_selections: list[dict[str, str]]) -> SpeakerProfile:
        selections = voice_selections or [
            {"voice_id": "voice_a", "role": "Host", "type": "preset", "name": "Alex"},
            {"voice_id": "voice_b", "role": "Guest", "type": "preset", "name": "Riley"},
        ]
        speakers = [
            Speaker(
                name=selection.get("name", selection["voice_id"]),
                voice_id=selection["voice_id"],
                role=selection["role"],
                type=selection.get("type", "preset"),
            )
            for selection in selections
        ]
        return SpeakerProfile(speakers=speakers)

    def mock_generate_outline(
        topic: str,
        key_points: list[str],
        briefing: str,
        num_segments: int,
        speakers: SpeakerProfile,
    ) -> Outline:
        segments = [
            Segment(
                title=f"Segment {index + 1}",
                description="Mock description.",
                size="short" if index % 2 == 0 else "medium",
            )
            for index in range(num_segments)
        ]
        return Outline(segments=segments)

    def mock_generate_transcript(
        outline: Outline,
        topic: str,
        briefing: str,
        speakers: SpeakerProfile,
        personas: dict[str, Persona] | None = None,
    ) -> Transcript:
        dialogues = []
        speaker_names = [speaker.name for speaker in speakers.speakers]
        for index, segment in enumerate(outline.segments):
            speaker_name = speaker_names[index % len(speaker_names)]
            dialogues.append(
                {
                    "speaker": speaker_name,
                    "text": f"Mock line for {segment.title}.",
                }
            )
        return Transcript.model_validate({"dialogues": dialogues})

    def mock_generate_all_clips(
        transcript: Transcript,
        speaker_profile: SpeakerProfile,
        params: dict[str, object],
        clips_dir: str | Path,
        progress_callback: Callable[[int, int, dict[str, object]], None] | None = None,
    ) -> list[str]:
        clips_dir = Path(clips_dir)
        clips_dir.mkdir(parents=True, exist_ok=True)
        total = len(transcript.dialogues)
        paths: list[str] = []
        for idx, dialogue in enumerate(transcript.dialogues):
            filename = f"{idx:04d}.mp3"
            path = clips_dir / filename
            _ = path.write_bytes(b"ID3")
            segment_info = {
                "index": idx,
                "speaker": dialogue.speaker,
                "text": dialogue.text,
                "filename": filename,
                "status": "success",
                "error": None,
                "path": str(path),
            }
            if progress_callback is not None:
                progress_callback(idx + 1, total, segment_info)
            paths.append(str(path))
        return paths

    def mock_combine_audio_clips(
        clips_dir: Path | str,
        output_path: Path | str,
        bitrate: str = "192k",
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _ = output_path.write_bytes(b"ID3")
        return output_path

    import config
    config.get_openai_api_key = lambda: "test"
    voice_selection.create_speaker_profile = mock_create_profile
    outline_generator.generate_outline = mock_generate_outline
    transcript_generator.generate_transcript = mock_generate_transcript
    batch_processor.generate_all_clips = mock_generate_all_clips
    audio_combiner.combine_audio_clips = mock_combine_audio_clips

    mock_content = {
        "topic": "The future of AI in creative industries",
        "key_points": "- History\n- Tools\n- Ethics",
        "briefing": "Conversational and insightful.",
        "num_segments": 3,
    }
    mock_voices = [
        {"voice_id": "voice_a", "role": "Host", "type": "preset", "name": "Alex"},
        {"voice_id": "voice_b", "role": "Guest", "type": "preset", "name": "Riley"},
    ]

    artifacts = generate_podcast(
        content_input=mock_content,
        voice_selections=mock_voices,
        quality_preset="draft",
        progress_callback=mock_progress,
    )
    print("Generated artifacts:")
    for key, value in artifacts.items():
        print(f"  {key}: {value}")
