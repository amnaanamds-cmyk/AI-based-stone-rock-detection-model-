"""Preprocessing: one dataset area -> model-ready training samples.

The image is processed with exactly the same functions the application uses at prediction time:

1. ``rockmap.pipeline.load_scene``  - scale pixel values to reflectance (sensor preset), mask cloud /
   no-data, map snow, water, dense vegetation and deep shadow (never used as rock samples)
2. ``rockmap.features.build_features`` - 6 bands + 7 spectral indices (+ 5 terrain features with a DEM)
3. ``rockmap.pipeline.SampleCollector`` - stratified samples per class from square blocks
   (train / validation / test), pixel vectors for RF / SVM and 9 x 9 patches for the CNN

The feature list, sensor preset, masking switch and the normaliser (mean / std per feature) are stored
in the model's ``meta.json``; the prediction module reads them back, so training and prediction can
never drift apart. Samples are cached in ``dataset/processed`` and reused while the area's files and
the sampling settings stay the same.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

from rockmap.config import CLASS_IDS
from rockmap.features import build_features
from rockmap.pipeline import SampleCollector, load_labels, load_scene
from rockmap.preprocessing import detect_sensor

from .dataset import Area, class_id_for


def sensor_for(area: Area, default: str = "auto") -> str:
    sensor = area.settings.get("sensor", default)
    return detect_sensor(area.image) if sensor == "auto" else sensor


def label_array(area: Area, info, work_dir: Path) -> np.ndarray:
    """Reference labels on the image grid (vector maps are rasterised first)."""
    if area.labels_kind == "raster":
        return load_labels(area.labels, info)
    from rockmap.reference import rasterize_reference
    field_name = area.settings.get("label_field", "class_id")
    raw = {str(k): v for k, v in area.settings.get("mapping", {}).items()}
    fc = json.loads(area.labels.read_text(encoding="utf-8"))
    values = {str((f.get("properties") or {}).get(field_name)) for f in fc.get("features", [])}
    mapping = {v: class_id_for(v, raw) for v in values}
    mapping = {k: v for k, v in mapping.items() if v}
    work_dir.mkdir(parents=True, exist_ok=True)
    out = rasterize_reference(area.labels, area.image, work_dir / f"{area.name}_labels.tif", field_name, mapping)
    return load_labels(out, info)


def _cache_key(area: Area, settings: dict) -> str:
    h = hashlib.sha1(json.dumps(settings, sort_keys=True).encode())
    for p in area.files():
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()[:12]


def area_samples(area: Area, settings: dict, processed_dir: Path, use_cache: bool = True,
                 log=print) -> dict:
    """Samples of one area: {"splits": {name: (vectors, patches, labels)}, "feature_names", "sensor",
    "class_pixels"} - from cache when the area and settings are unchanged."""
    sensor = sensor_for(area, settings["default_sensor"])
    key = _cache_key(area, {**settings, "sensor": sensor})
    cache = Path(processed_dir) / f"{area.name}_{key}.npz"
    if use_cache and cache.exists():
        z = np.load(cache, allow_pickle=False)
        info = json.loads(str(z["info"]))
        splits = {s: (z[f"{s}_x"], z[f"{s}_p"] if f"{s}_p" in z else None, z[f"{s}_y"]) for s in SampleCollector.SPLITS}
        log(f"  {area.name}: cached samples ({cache.name})")
        return {**info, "splits": splits}

    use_dem = settings["use_dem"]
    scene = load_scene(area.image, area.dem if use_dem else None, sensor=sensor, masking=settings["masking"])
    labels = label_array(area, scene.info, Path(processed_dir))
    labels[~np.isin(labels, CLASS_IDS)] = 0
    features, names = build_features(scene.reflectance, scene.dem if use_dem else None, scene.pixel_size_m)
    usable = scene.usable
    class_pixels = {int(c): int(((labels == c) & usable).sum()) for c in CLASS_IDS if ((labels == c) & usable).any()}
    if not class_pixels:
        raise ValueError(f"{area.name}: no labelled pixels left after masking (snow, cloud, water, vegetation)")
    seed = settings["seed"] + int(hashlib.sha1(area.name.encode()).hexdigest()[:6], 16) % 10_000
    col = SampleCollector(settings["samples_per_class"], settings["block_size"], patches=settings["patches"],
                          seed=seed, fractions=settings["split"])
    col.add(features, names, labels, usable)
    splits = {}
    for s in SampleCollector.SPLITS:
        parts = col.parts[s]
        if parts:
            splits[s] = (np.concatenate([p[0] for p in parts]),
                         np.concatenate([p[1] for p in parts]) if settings["patches"] else None,
                         np.concatenate([p[2] for p in parts]))
        else:
            splits[s] = (np.zeros((0, len(names)), np.float32), None, np.zeros(0, np.int64))
    info = {"feature_names": names, "sensor": sensor, "class_pixels": class_pixels,
            "pixel_size_m": float(scene.pixel_size_m)}
    if use_cache:
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        for old in Path(processed_dir).glob(f"{area.name}_*.npz"):
            old.unlink()
        arrays = {"info": np.array(json.dumps(info))}
        for s, (x, p, y) in splits.items():
            arrays[f"{s}_x"], arrays[f"{s}_y"] = x, y
            if p is not None:
                arrays[f"{s}_p"] = p
        np.savez_compressed(cache, **arrays)
    log(f"  {area.name}: {sum(len(v[2]) for v in splits.values()):,} samples from "
        f"{sum(class_pixels.values()):,} labelled pixels ({sensor})")
    return {**info, "splits": splits}


def pooled_collector(per_area: list[dict], settings: dict) -> SampleCollector:
    """Merge the samples of all areas into one collector (per-class caps are applied after pooling)."""
    names = per_area[0]["feature_names"]
    for a in per_area[1:]:
        if a["feature_names"] != names:
            raise ValueError("All areas must provide the same features - give every area a dem.tif or none")
    col = SampleCollector(settings["samples_per_class"], settings["block_size"], patches=settings["patches"],
                          seed=settings["seed"], fractions=settings["split"])
    col.feature_names = list(names)
    for a in per_area:
        for s, (x, p, y) in a["splits"].items():
            if len(y):
                col.parts[s].append((x, p, y))
    col.n_sources = len(per_area)
    return col


def resolve_use_dem(areas: list[Area], setting) -> bool:
    if setting == "auto":
        return all(a.dem is not None for a in areas)
    return bool(setting)


def describe_classes(class_pixels: dict[int, int], names: dict[int, str]) -> str:
    return "\n".join(f"    {cid}  {names.get(cid, cid):34s} {n:>10,} labelled pixels"
                     for cid, n in sorted(class_pixels.items()))

