"""Find and check the study areas in the dataset folder.

Expected layout (full explanation in dataset/README.md)::

    dataset/raw/
        gilgit_2025/
            image.tif          multispectral image, 6 bands: blue, green, red, NIR, SWIR1, SWIR2
            labels.tif         reference map: one band, class id per pixel (0 = unlabelled)
              - or -
            labels.geojson     geological map polygons + area.json telling which field holds the unit
            dem.tif            optional elevation model (any resolution / projection)
            area.json          optional settings (sensor, label field, unit -> class mapping)
        hunza_field_2026/
            ...

Every sub-folder that contains an image is one "area". All areas are pooled for training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio

from rockmap.config import CLASS_BY_ID, CLASS_IDS, ROCK_CLASSES

IMAGE_NAMES = ("image.tif", "image.tiff")
LABEL_RASTER_NAMES = ("labels.tif", "labels.tiff")
LABEL_VECTOR_NAMES = ("labels.geojson", "labels.json")
DEM_NAMES = ("dem.tif", "dem.tiff")
SETTINGS_NAME = "area.json"
SAMPLE_MARKER = "SAMPLE_DATA.txt"      # present in generated sample data (not real geology)


@dataclass
class Area:
    """One study area: an image, its reference labels and optional DEM / settings."""
    name: str
    folder: Path
    image: Path
    labels: Optional[Path]
    labels_kind: Optional[str]          # "raster" or "vector"
    dem: Optional[Path]
    settings: dict = field(default_factory=dict)

    @property
    def is_sample(self) -> bool:
        return (self.folder / SAMPLE_MARKER).exists()

    def files(self) -> list[Path]:
        return [p for p in (self.image, self.labels, self.dem, self.folder / SETTINGS_NAME) if p and p.exists()]


def _first(folder: Path, names) -> Optional[Path]:
    for n in names:
        if (folder / n).exists():
            return folder / n
    return None


def _area(folder: Path) -> Optional[Area]:
    image = _first(folder, IMAGE_NAMES)
    if image is None:
        return None
    raster, vector = _first(folder, LABEL_RASTER_NAMES), _first(folder, LABEL_VECTOR_NAMES)
    settings = {}
    if (folder / SETTINGS_NAME).exists():
        settings = json.loads((folder / SETTINGS_NAME).read_text(encoding="utf-8"))
    return Area(folder.name, folder, image, raster or vector,
                "raster" if raster else "vector" if vector else None, _first(folder, DEM_NAMES), settings)


def find_areas(data_dir: Path) -> list[Area]:
    """All areas in ``data_dir`` (sub-folders with an image), or ``data_dir`` itself if it is one area."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    single = _area(data_dir)
    if single:
        return [single]
    return [a for a in (_area(p) for p in sorted(data_dir.iterdir()) if p.is_dir()) if a]


def class_id_for(value, mapping: dict) -> Optional[int]:
    """Translate a map unit to a class id using ``mapping`` (values may be ids or class names)."""
    target = mapping.get(str(value), value if not mapping else None)
    if target is None:
        return None
    if isinstance(target, str) and not target.isdigit():
        by_name = {c.name.lower(): c.id for c in ROCK_CLASSES}
        short = {c.name.split(" / ")[0].split(" (")[0].lower(): c.id for c in ROCK_CLASSES}
        return by_name.get(target.lower()) or short.get(target.lower())
    return int(target) if int(target) in CLASS_BY_ID else None


def check_area(area: Area) -> list[str]:
    """Problems that would stop training on this area (empty list = OK)."""
    problems = []
    try:
        with rasterio.open(area.image) as src:
            if src.count < 6:
                problems.append(f"image.tif has {src.count} bands; 6 are needed (blue, green, red, NIR, SWIR1, SWIR2)")
            if src.crs is None:
                problems.append("image.tif has no coordinate system (CRS); export it as a GeoTIFF")
            img_bounds, img_crs = src.bounds, src.crs
    except rasterio.RasterioIOError as e:
        return [f"cannot read image.tif: {e}"]
    if area.labels is None:
        problems.append("no labels.tif or labels.geojson found")
    elif area.labels_kind == "raster":
        with rasterio.open(area.labels) as src:
            sample = src.read(1, out_shape=(min(src.height, 512), min(src.width, 512)))
            if src.crs and img_crs and src.crs != img_crs:
                pass                       # reprojected automatically
            from rasterio.warp import transform_bounds
            b = transform_bounds(src.crs or img_crs, img_crs, *src.bounds) if src.crs else src.bounds
            if b[2] <= img_bounds[0] or b[0] >= img_bounds[2] or b[3] <= img_bounds[1] or b[1] >= img_bounds[3]:
                problems.append("labels.tif does not overlap image.tif")
        values = set(np.unique(sample).tolist()) - {0}
        bad = sorted(v for v in values if v not in CLASS_IDS and v < 250)
        if bad:
            problems.append(f"labels.tif contains unknown class ids {bad} (valid: {CLASS_IDS}, 0 = unlabelled)")
        if not values:
            problems.append("labels.tif contains no labelled pixels (all 0)")
    else:
        field_name = area.settings.get("label_field", "class_id")
        try:
            fc = json.loads(area.labels.read_text(encoding="utf-8"))
        except ValueError as e:
            return problems + [f"labels.geojson is not valid JSON: {e}"]
        feats = fc.get("features", [])
        if not feats:
            problems.append("labels.geojson has no features")
        mapping = {str(k): v for k, v in area.settings.get("mapping", {}).items()}
        values = {(f.get("properties") or {}).get(field_name) for f in feats}
        mapped = [v for v in values if class_id_for(v, mapping)]
        if feats and not mapped:
            problems.append(f"no polygon value in field '{field_name}' maps to a class - set \"label_field\" and "
                            f"\"mapping\" in area.json (see dataset/README.md)")
    if area.dem is not None:
        try:
            with rasterio.open(area.dem):
                pass
        except rasterio.RasterioIOError as e:
            problems.append(f"cannot read dem.tif: {e}")
    return problems


def describe(areas: list[Area]) -> str:
    lines = []
    for a in areas:
        with rasterio.open(a.image) as src:
            size = f"{src.width} x {src.height} px, {src.count} bands, {abs(src.transform.a):g} m"
        extra = [a.labels_kind + " labels" if a.labels_kind else "NO LABELS", "DEM" if a.dem else "no DEM"]
        if a.is_sample:
            extra.append("SAMPLE DATA (synthetic, not real geology)")
        lines.append(f"  - {a.name}: {size}; " + ", ".join(extra))
    return "\n".join(lines)
