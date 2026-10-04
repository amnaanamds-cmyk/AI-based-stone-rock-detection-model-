"""Where trained models live and which one is "current".

Shared by the training module (which writes new versions), the prediction module and the web
application (which only read). Layout::

    models/trained/
        model_v1/          one complete model bundle per training run (never overwritten)
            meta.json      classes, preprocessing parameters, metrics, training metadata
            cnn.pt  rf.joblib  svm.joblib  cm_<algo>.png  evaluation.txt  training_config.json
        model_v2/
        current.json       {"version": "model_v2", ...}  <- the model the application uses

The folder can be moved with the ``ROCKMAP_MODELS_DIR`` environment variable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

_VERSION = re.compile(r"^model_v(\d+)$")


def project_root() -> Path:
    """The repository root (the folder that contains ``rockmap/``, ``training/`` and ``models/``)."""
    return Path(__file__).resolve().parents[1]


def models_dir() -> Path:
    env = os.environ.get("ROCKMAP_MODELS_DIR")
    return Path(env) if env else project_root() / "models" / "trained"


def versions(root: Optional[Path] = None) -> list[str]:
    """Model versions present on disk, oldest first (``model_v1``, ``model_v2`` ...)."""
    root = Path(root) if root else models_dir()
    if not root.exists():
        return []
    found = [(int(m.group(1)), p.name) for p in root.iterdir()
             if p.is_dir() and (m := _VERSION.match(p.name)) and (p / "meta.json").exists()]
    return [name for _, name in sorted(found)]


def next_version_dir(root: Optional[Path] = None) -> Path:
    root = Path(root) if root else models_dir()
    root.mkdir(parents=True, exist_ok=True)
    nums = [int(_VERSION.match(p.name).group(1)) for p in root.iterdir() if p.is_dir() and _VERSION.match(p.name)]
    return root / f"model_v{max(nums, default=0) + 1}"


def current_version(root: Optional[Path] = None) -> Optional[str]:
    """Name of the model the application uses, or None when no model has been trained yet."""
    root = Path(root) if root else models_dir()
    p = root / "current.json"
    if p.exists():
        try:
            name = json.loads(p.read_text(encoding="utf-8")).get("version")
            if name and (root / name / "meta.json").exists():
                return name
        except (OSError, ValueError):
            pass
    found = versions(root)                    # no pointer (or a broken one): newest complete version
    return found[-1] if found else None


def current_model_path(root: Optional[Path] = None) -> Optional[Path]:
    root = Path(root) if root else models_dir()
    name = current_version(root)
    return root / name if name else None


def set_current(version: str, root: Optional[Path] = None) -> Path:
    root = Path(root) if root else models_dir()
    if not (root / version / "meta.json").exists():
        raise ValueError(f"{version} is not a complete model in {root} (have: {', '.join(versions(root)) or 'none'})")
    p = root / "current.json"
    tmp = root / "current.json.tmp"
    tmp.write_text(json.dumps({"version": version, "activated": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2),
                   encoding="utf-8")
    tmp.replace(p)
    return root / version


def read_meta(version: str, root: Optional[Path] = None) -> dict:
    root = Path(root) if root else models_dir()
    return json.loads((root / version / "meta.json").read_text(encoding="utf-8"))


def delete_version(version: str, root: Optional[Path] = None) -> None:
    root = Path(root) if root else models_dir()
    if version == current_version(root):
        raise ValueError(f"{version} is the current model - activate another version first")
    if not _VERSION.match(version) or not (root / version).is_dir():
        raise ValueError(f"no model version {version}")
    shutil.rmtree(root / version)
