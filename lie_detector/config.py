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
    av_understanding_dir: Path = field(
        default_factory=lambda: _env_path(
            "AV_UNDERSTANDING_DIR",
            Path("/Users/karan/Desktop/work/whissle/live_assist/Audio-visual-understanding"),
        )
    )
    whissle_python_api_dir: Path = field(
        default_factory=lambda: _env_path(
            "WHISSLE_PYTHON_API_DIR",
            Path("/Users/karan/Desktop/work/whissle/live_assist/whissle_python_api"),
        )
    )

    # --- Whissle STT -------------------------------------------------------
    whissle_api_token: str = field(default_factory=lambda: _env("WHISSLE_API_TOKEN"))
    whissle_auth_token: str = field(default_factory=lambda: _env("WHISSLE_AUTH_TOKEN"))
    whissle_mode: str = field(default_factory=lambda: _env("WHISSLE_MODE", "gateway").lower())
    whissle_asr_model: str = field(default_factory=lambda: _env("WHISSLE_ASR_MODEL", "en-US-NER"))

    # --- Visual lane -------------------------------------------------------
    visual_sample_fps: float = field(default_factory=lambda: float(_env("VISUAL_SAMPLE_FPS", "5") or 5))
    visual_semantic_lane: bool = field(default_factory=lambda: _env_bool("VISUAL_SEMANTIC_LANE", False))
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-4-5"))

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
    def stt_dir(self) -> Path:
        return self.data_dir / "stt"

    @property
    def visual_dir(self) -> Path:
        return self.data_dir / "visual"

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
            self.data_dir, self.wav_dir, self.stt_dir, self.visual_dir,
            self.audio_dir, self.features_dir, self.models_dir, self.reports_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


CFG = Config()
