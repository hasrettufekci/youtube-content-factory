"""
Per-video project folder (spec section 25; extended for multilingual
production in Round 3 section 10):

    storage/2026/09/mysterious-place/
        research.json
        script_en.json          # canonical/master script
        script_tr.json          # localization(s) - one file per language
        scenes.json             # SHARED scene plan (visual_prompt is
                                 # language-independent; narration_en/
                                 # narration_tr live on each scene)
        scene_001.png ... scene_NNN.png   # SHARED visual production -
                                 # generated exactly once, reused by every
                                 # language's audio track
        narration_en.mp3        # per-language narration audio
        narration_tr.mp3
        video.mp4                # ONE video: shared visuals + the master
                                 # (English) narration track
        thumbnail.png            # shared, language-neutral
        metadata_en.json
        metadata_tr.json
        state.json

Every stage of the pipeline reads/writes only through this module, so a
failure at (say) video assembly never has to touch research/script/images
again - `pipeline.py` resumes by checking which files already exist.

Backward compatibility: every pre-Round-3 property (`script_json`,
`narration_mp3`, `metadata_json`) is kept as a plain alias for that same
artifact's MASTER-language path (`script_en_json`, `narration_en_mp3`,
`metadata_en_json`). They point at the exact same file, so every caller
and test written before multilingual support keeps working unchanged -
the physical filename on disk is simply now `script_en.json` instead of
`script.json`, etc.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from core.state import PipelineState
from storage import db


def slugify(text: str, max_words: int = 6) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    words = text.split()[:max_words]
    slug = "-".join(words).strip("-")
    return slug or "untitled"


def normalize_topic(topic: str) -> str:
    """Loose normalization used for duplicate-topic detection (spec 28)."""
    text = unicodedata.normalize("NFKD", topic).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(text.split())


@dataclass
class ProjectPaths:
    video_id: int
    slug: str
    root: Path

    @property
    def research_json(self) -> Path:
        return self.root / "research.json"

    # --- Master (English) script. `script_json` is kept as a plain alias
    # so every pre-Round-3 caller/test keeps working unchanged - the file
    # that actually lands on disk is named script_en.json.
    @property
    def script_en_json(self) -> Path:
        return self.root / "script_en.json"

    @property
    def script_json(self) -> Path:
        return self.script_en_json

    def script_localized_json(self, language: str) -> Path:
        """script_<lang>.json for any localization language (Round 3
        sections 4/9), e.g. script_tr.json. Not for the master language -
        use script_en_json/script_json for that. Generic in `language` so
        adding a future localization (German, Spanish, ...) needs no new
        method here."""
        return self.root / f"script_{language}.json"

    @property
    def script_tr_json(self) -> Path:
        return self.script_localized_json("tr")

    @property
    def scenes_json(self) -> Path:
        return self.root / "scenes.json"

    # --- Narration audio: one file per language (Round 3 section 6).
    # `narration_mp3` is kept as an alias for the master (English) track.
    def narration_mp3_for(self, language: str) -> Path:
        return self.root / f"narration_{language}.mp3"

    @property
    def narration_en_mp3(self) -> Path:
        return self.narration_mp3_for("en")

    @property
    def narration_tr_mp3(self) -> Path:
        return self.narration_mp3_for("tr")

    @property
    def narration_mp3(self) -> Path:
        return self.narration_en_mp3

    @property
    def video_mp4(self) -> Path:
        return self.root / "video.mp4"

    def video_mp4_for(self, language: str) -> Path:
        """The primary pipeline only ever produces `video_mp4` (shared
        visuals + the master narration track - spec Round 3 section 2/14).
        This is an architecture-readiness hook for a future/optional
        per-language render (still reusing the exact same scene images,
        never regenerating them) - not called anywhere in the default
        pipeline flow today."""
        return self.video_mp4 if language == settings.LANGUAGE_MASTER else self.root / f"video_{language}.mp4"

    @property
    def thumbnail_png(self) -> Path:
        return self.root / "thumbnail.png"

    # --- Metadata: one file per language (Round 3 section 15).
    # `metadata_json` is kept as an alias for the master (English) file.
    def metadata_json_for(self, language: str) -> Path:
        return self.root / f"metadata_{language}.json"

    @property
    def metadata_en_json(self) -> Path:
        return self.metadata_json_for("en")

    @property
    def metadata_tr_json(self) -> Path:
        return self.metadata_json_for("tr")

    @property
    def metadata_json(self) -> Path:
        return self.metadata_en_json

    # --- Subtitles: data-model/interface stub only (Round 3 section 16) -
    # no generation code calls this yet; it exists so adding real subtitle
    # export later is a new function writing to an already-defined path,
    # not a schema change.
    def subtitle_path_for(self, language: str, ext: str = "srt") -> Path:
        return self.root / f"subtitles_{language}.{ext}"

    @property
    def state_json(self) -> Path:
        return self.root / "state.json"

    def scene_image(self, index: int) -> Path:
        return self.root / f"scene_{index:03d}.png"

    def scene_images(self) -> list[Path]:
        return sorted(self.root.glob("scene_*.png"))

    # --- Shorts Engine V1 (additive): one vertical (9:16) video + thumbnail
    # + narration per Short, stored under this Long's own project folder
    # (spec's conceptual storage/<slug>/shorts/short_0N.* layout) - reuses
    # this SAME ProjectPaths/root convention rather than inventing a
    # parallel storage abstraction. Generic in `short_number` so
    # SHORTS_PER_LONG can grow past 3 with zero code change here.
    def short_dir(self) -> Path:
        return self.root / "shorts"

    def short_video_path(self, short_number: int) -> Path:
        return self.short_dir() / f"short_{short_number:02d}.mp4"

    def short_thumbnail_path(self, short_number: int) -> Path:
        return self.short_dir() / f"short_{short_number:02d}.png"

    def short_narration_path(self, short_number: int) -> Path:
        return self.short_dir() / f"short_{short_number:02d}_narration.mp3"

    def short_scene_image(self, short_number: int) -> Path:
        """The (possibly regenerated-vertical, possibly cropped) hero image
        used to build this Short's video - kept alongside its other
        artifacts rather than named scene_NNN.png, which is reserved for
        the Long's own shared 16:9 scene images (never touched by Shorts
        production)."""
        return self.short_dir() / f"short_{short_number:02d}_visual.png"


def _paths_for(video_id: int, slug: str, storage_path: str) -> ProjectPaths:
    return ProjectPaths(video_id=video_id, slug=slug, root=Path(storage_path))


def create_project(
    topic: str,
    category: str | None = None,
    project_id: str | None = None,
    force_new: bool = False,
) -> tuple[ProjectPaths, bool]:
    """Create (or, unless force_new, reuse) the on-disk project folder + DB
    row for a topic. Returns (paths, created) - created=False means an
    existing video with the same normalized topic was reused (idempotency,
    spec section 28)."""
    project_id = project_id or settings.DEFAULT_PROJECT_ID
    normalized = normalize_topic(topic)

    if not force_new:
        existing = db.find_video_by_normalized_topic(normalized)
        if existing is not None:
            return _paths_for(existing["id"], existing["slug"], existing["storage_path"]), False

    now = datetime.now(timezone.utc)
    base_slug = slugify(topic)
    slug = base_slug
    suffix = 1
    while db.get_video_by_slug(slug) is not None:
        suffix += 1
        slug = f"{base_slug}-{suffix}"

    root = settings.STORAGE_DIR / f"{now:%Y}" / f"{now:%m}" / slug
    root.mkdir(parents=True, exist_ok=True)

    video_id = db.insert_video(
        slug=slug,
        topic=topic,
        normalized_topic=normalized,
        category=category,
        project_id=project_id,
        state=str(PipelineState.TOPIC_CREATED),
        storage_path=str(root),
    )

    paths = ProjectPaths(video_id=video_id, slug=slug, root=root)
    save_state(paths, PipelineState.TOPIC_CREATED, extra={"topic": topic, "category": category})
    return paths, True


def get_project(video_id: int) -> ProjectPaths | None:
    row = db.get_video(video_id)
    if row is None:
        return None
    return _paths_for(row["id"], row["slug"], row["storage_path"])


def get_project_by_slug(slug: str) -> ProjectPaths | None:
    row = db.get_video_by_slug(slug)
    if row is None:
        return None
    return _paths_for(row["id"], row["slug"], row["storage_path"])


# --- generic JSON helpers -------------------------------------------------

def save_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_json_atomic(path: Path, data: dict | list) -> None:
    """Same as save_json(), but write-to-temp-file-then-rename (mirrors
    youtube/image_gen.py's scene image writes) - `Path.replace` is an
    atomic rename on both POSIX and Windows, so a process killed mid-write
    never leaves a partially-written/corrupt file at `path`. Used where an
    existing, otherwise-intact shared artifact (e.g. scenes.json, being
    enriched in place rather than freshly created) must never be
    corrupted by an interrupted write (Round 3.1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- state.json + DB state kept in sync ----------------------------------

def save_state(paths: ProjectPaths, state: PipelineState, extra: dict | None = None) -> None:
    current = load_json(paths.state_json) or {}
    if not isinstance(current, dict):
        current = {}
    current["state"] = str(state)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    if extra:
        current.update(extra)
    save_json(paths.state_json, current)
    db.update_video_state(paths.video_id, str(state), error_message=current.get("error"))


def save_failure(paths: ProjectPaths, failed_stage: str, error: str) -> None:
    current = load_json(paths.state_json) or {}
    if not isinstance(current, dict):
        current = {}
    current["state"] = str(PipelineState.FAILED)
    current["failed_stage"] = failed_stage
    current["error"] = error
    current["updated_at"] = datetime.now(timezone.utc).isoformat()
    save_json(paths.state_json, current)
    db.update_video_state(paths.video_id, str(PipelineState.FAILED), error_message=f"[{failed_stage}] {error}")


def load_state(paths: ProjectPaths) -> PipelineState:
    data = load_json(paths.state_json)
    if not data or "state" not in data:
        return PipelineState.TOPIC_CREATED
    try:
        return PipelineState(data["state"])
    except ValueError:
        return PipelineState.TOPIC_CREATED
