"""
Pipeline state machine (spec section 26).

A video moves forward through ORDER one step at a time. Any state can move
to FAILED. FAILED can be retried back into the state it failed *from*
(the pipeline re-runs only that stage - see pipeline.py's resume logic -
so a broken video-assembly step never forces research/images to redo).
"""
from __future__ import annotations

from enum import Enum


class PipelineState(str, Enum):
    TOPIC_CREATED = "TOPIC_CREATED"
    RESEARCHING = "RESEARCHING"
    RESEARCHED = "RESEARCHED"
    SCRIPT_GENERATED = "SCRIPT_GENERATED"
    SCENES_PLANNED = "SCENES_PLANNED"
    IMAGES_GENERATED = "IMAGES_GENERATED"
    AUDIO_GENERATED = "AUDIO_GENERATED"
    VIDEO_ASSEMBLED = "VIDEO_ASSEMBLED"
    THUMBNAIL_READY = "THUMBNAIL_READY"
    PENDING_APPROVAL = "PENDING_APPROVAL"
    APPROVED = "APPROVED"
    UPLOADING = "UPLOADING"
    PUBLISHED = "PUBLISHED"
    FAILED = "FAILED"

    def __str__(self) -> str:  # nicer logs / DB storage
        return self.value


# Linear happy-path order. Index used to know "what comes next" and to
# validate that we never silently skip a stage.
ORDER: list[PipelineState] = [
    PipelineState.TOPIC_CREATED,
    PipelineState.RESEARCHING,
    PipelineState.RESEARCHED,
    PipelineState.SCRIPT_GENERATED,
    PipelineState.SCENES_PLANNED,
    PipelineState.IMAGES_GENERATED,
    PipelineState.AUDIO_GENERATED,
    PipelineState.VIDEO_ASSEMBLED,
    PipelineState.THUMBNAIL_READY,
    PipelineState.PENDING_APPROVAL,
    PipelineState.APPROVED,
    PipelineState.UPLOADING,
    PipelineState.PUBLISHED,
]

_INDEX = {state: i for i, state in enumerate(ORDER)}

TERMINAL_STATES = {PipelineState.PUBLISHED, PipelineState.FAILED}


def next_state(current: PipelineState) -> PipelineState | None:
    """State that should follow `current` on success, or None if terminal."""
    if current not in _INDEX:
        return None
    i = _INDEX[current]
    if i + 1 >= len(ORDER):
        return None
    return ORDER[i + 1]


def can_advance(current: PipelineState, target: PipelineState) -> bool:
    """True if moving from `current` straight to `target` is a legal
    forward step (one stage at a time) or a move into FAILED (always legal)."""
    if target == PipelineState.FAILED:
        return True
    if current not in _INDEX or target not in _INDEX:
        return False
    return _INDEX[target] == _INDEX[current] + 1


def stage_that_produces(state: PipelineState) -> str:
    """Human label of the pipeline stage responsible for reaching `state` -
    used in retry/resume logging so failures say what will be re-run."""
    labels = {
        PipelineState.RESEARCHING: "research",
        PipelineState.RESEARCHED: "research",
        PipelineState.SCRIPT_GENERATED: "script",
        PipelineState.SCENES_PLANNED: "scene_plan",
        PipelineState.IMAGES_GENERATED: "images",
        PipelineState.AUDIO_GENERATED: "narration",
        PipelineState.VIDEO_ASSEMBLED: "video_assembly",
        PipelineState.THUMBNAIL_READY: "thumbnail",
        PipelineState.PENDING_APPROVAL: "telegram_approval",
        PipelineState.APPROVED: "approval",
        PipelineState.UPLOADING: "youtube_upload",
        PipelineState.PUBLISHED: "youtube_upload",
    }
    return labels.get(state, state.value.lower())
