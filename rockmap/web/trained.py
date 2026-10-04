"""Make the training module's current model available in the web application.

The application never trains on start-up. It only reads ``models/trained/current.json`` (written by
``python training/train.py`` or ``python training/models.py use ...``) and registers that model
version in its model list, so it can be chosen for scenes and regions. The check is repeated every
few seconds, so a newly trained or re-activated version appears without restarting the server.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .. import model_store
from .db import Database

SOURCE = "training_module"
_state: dict = {}           # per database: last check time and the model file version seen


def model_summary(meta: dict) -> tuple[dict, str]:
    summary = {a: {"label": m["label"], "oa": m["test_metrics"]["overall_accuracy"],
                   "kappa": m["test_metrics"]["kappa"], "f1": m["test_metrics"]["macro_f1"],
                   "seconds": m["train_seconds"]} for a, m in meta["algorithms"].items()}
    best = meta.get("app_algorithm") or max(summary, key=lambda a: summary[a]["kappa"])
    return summary, best


def sync_trained_model(db: Database, force: bool = False, every: float = 5.0) -> Optional[dict]:
    """Register the current training-module model in the ``models`` table; returns its row."""
    now = time.time()
    _last = _state.setdefault(str(getattr(db, "path", id(db))), {"t": 0.0, "key": None})
    if not force and now - _last["t"] < every:
        return None
    _last["t"] = now
    path = model_store.current_model_path()
    rows = db.all("models", "source = ?", (SOURCE,))
    if path is None:
        return None
    meta_file = path / "meta.json"
    key = (str(path), meta_file.stat().st_mtime)
    row = next((r for r in rows if r.get("version") == path.name and Path(r["folder"]) == path), None)
    for r in rows:                       # only the active version keeps the "(current)" mark
        if r is not row and r["name"].endswith("(current)"):
            db.update("models", r["id"], name=r["name"][:-len(" (current)")])
    if row and _last["key"] == key:
        return row
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    summary, best = model_summary(meta)
    name = f"{path.name} - training module{' [SAMPLE DATA]' if meta.get('sample_data') else ''} (current)"
    if row:
        db.update("models", row["id"], name=name, summary=summary, best_algo=best, uses_dem=int(meta["uses_dem"]))
    else:
        db.insert("models", name=name, folder=str(path), uses_dem=int(meta["uses_dem"]), best_algo=best,
                  summary=summary, created_by="training/train.py", source=SOURCE, version=path.name)
    _last["key"] = key
    return db.all("models", "source = ? AND version = ?", (SOURCE, path.name))[0]


def current_real_model(db: Database) -> Optional[dict]:
    """The current training-module model, unless it was trained only on synthetic sample data."""
    row = sync_trained_model(db, force=True)
    if not row:
        return None
    meta = json.loads((Path(row["folder"]) / "meta.json").read_text(encoding="utf-8"))
    return None if meta.get("sample_data") else row


def is_training_module(model: dict) -> bool:
    return (model or {}).get("source") == SOURCE
