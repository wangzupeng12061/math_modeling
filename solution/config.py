"""Configuration helpers for the solution package."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    datasets_dir: Path
    outputs_dir: Path
    results_dir: Path


def _discover_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_settings() -> Settings:
    root = _discover_root()
    return Settings(
        root_dir=root,
        datasets_dir=root / "datasets",
        outputs_dir=root / "outputs",
        results_dir=root / "results",
    )


settings = load_settings()

__all__ = ["Settings", "settings", "load_settings"]
