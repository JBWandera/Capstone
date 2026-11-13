"""Project paths.

On Colab the data lives on Google Drive so it survives runtime resets.
Set AFROVIEWZ_DATA to override, for example on a local machine.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def data_root() -> Path:
    env = os.environ.get("AFROVIEWZ_DATA")
    if env:
        return Path(env)
    colab_drive = Path("/content/drive/MyDrive/AfroViewz/data")
    if colab_drive.parent.parent.exists():
        return colab_drive
    return REPO_ROOT / "data"


def raw_root() -> Path:
    """Raw downloads are large (Stellenbosch alone is about 23 GB), so on Colab they
    go to the fast local disk, not Drive. Only the cleaned, downscaled copy in
    interim/ is kept on Drive."""
    env = os.environ.get("AFROVIEWZ_RAW")
    if env:
        return Path(env)
    if Path("/content").exists():
        return Path("/content/afroviewz_raw")
    return data_root() / "raw"


def raw_dir(source: str) -> Path:
    return raw_root() / source


def interim_dir(source: str) -> Path:
    return data_root() / "interim" / source


def processed_dir(name: str) -> Path:
    return data_root() / "processed" / name
