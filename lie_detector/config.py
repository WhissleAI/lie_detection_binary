"""Central configuration: paths, env, and pipeline constants.

Everything is env-driven (loaded from a local .env if present) so the project is
self-contained and reproducible.  Import ``CFG`` anywhere:

    from lie_detector.config import CFG
    print(CFG.dataset_dir, CFG.stt_dir)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional, but recommended
    from dotenv import load_dotenv

    # Load .env from the project root (parent of this package dir).
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(_PROJECT_ROOT / ".env")
except Exception:  # pragma: no cover - dotenv is optional
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    v = _env(name)
    return Path(v).expanduser() if v else default


@dataclass(frozen=True)
class Config:
    # --- repo / data roots -------------------------------------------------
    project_root: Path = _PROJECT_ROOT
    data_dir: Path = _PROJECT_ROOT / "data"

    dataset_dir: Path = field(
        default_factory=lambda: _env_path(
            "DECEPTION_DATASET_DIR",
            Path("/Users/karan/Downloads/Real-life_Deception_Detection_2016"),
        )
    )

    # --- Whissle gateway (local docker or cloud) ---------------------------
    # The gateway's POST /video/analyze returns BOTH the segmented transcript
    # (with emotion/intent/age/gender metadata + diarization) AND the per-frame
    # visual timeline — i.e. Whissle STT + audio-visual in one call.
    gateway_url: str = field(default_factory=lambda: _env("WHISSLE_GATEWAY_URL", "http://localhost:9000").rstrip("/"))
    whissle_api_token: str = field(default_factory=lambda: _env("WHISSLE_API_TOKEN"))
    # ASR model to request from /asr/transcribe (empty = gateway default).
    asr_model_id: str = field(default_factory=lambda: _env("WHISSLE_ASR_MODEL", "whissle-large"))
    metadata_tags: str = field(default_factory=lambda: _env("WHISSLE_METADATA_TAGS", "emotion,intent,age,gender"))
    diarization: bool = field(default_factory=lambda: _env_bool("WHISSLE_DIARIZATION", True))

    # --- Visual lane (sampling + optional Claude semantic lane on the server)
    visual_sample_fps: float = field(default_factory=lambda: float(_env("VISUAL_SAMPLE_FPS", "5") or 5))
    visual_semantic_lane: bool = field(default_factory=lambda: _env_bool("VISUAL_SEMANTIC_LANE", False))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))

    # --- Gemini (LLM video baseline) --------------------------------------
    gemini_api_key: str = field(default_factory=lambda: _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY"))
    gemini_model: str = field(default_factory=lambda: _env("GEMINI_MODEL", "gemini-2.5-pro"))

    # --- modeling ----------------------------------------------------------
    random_seed: int = 42

    # --- derived data sub-dirs --------------------------------------------
    @property
    def clips_dir(self) -> Path:
        return self.dataset_dir / "Clips"

    @property
    def transcription_dir(self) -> Path:
        return self.dataset_dir / "Transcription"

    @property
    def annotation_csv(self) -> Path:
        return self.dataset_dir / "Annotation" / "All_Gestures_Deceptive and Truthful.csv"

    @property
    def readme_txt(self) -> Path:
        return self.dataset_dir / "README.txt"

    @property
    def manifest_csv(self) -> Path:
        return self.data_dir / "manifest.csv"

    @property
    def wav_dir(self) -> Path:
        return self.data_dir / "wav"

    @property
    def av_dir(self) -> Path:
        # Fused gateway /video/analyze output (transcript + segments + visual timeline)
        return self.data_dir / "av"

    @property
    def gemini_dir(self) -> Path:
        return self.data_dir / "gemini"

    @property
    def gemini_reason_dir(self) -> Path:
        # Gemini reasoning over our extracted features (not raw video)
        return self.data_dir / "gemini_reason"

    @property
    def gemini_reason_v2_dir(self) -> Path:
        # v2: neutral, base-rate-anchored, symmetric-evidence prompt (debiased)
        return self.data_dir / "gemini_reason_v2"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def features_dir(self) -> Path:
        return self.data_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_dir / "reports"

    def ensure_dirs(self) -> None:
        for d in (
            self.data_dir, self.wav_dir, self.av_dir, self.gemini_dir,
            self.gemini_reason_dir, self.audio_dir, self.features_dir,
            self.models_dir, self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
