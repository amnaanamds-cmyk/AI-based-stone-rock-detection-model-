"""Preparing reference geological maps (vector polygons -> label raster on the scene grid)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from rasterio.features import rasterize
from rasterio.warp import transform_geom

from .config import CLASS_BY_ID
from .io import class_colormap, read_raster, write_raster


def _read_features(vector_path: Path) -> tuple[list[dict], Optional[str]]:
    """Return GeoJSON-like features and the source CRS (None = EPSG:4326 for GeoJSON)."""
    if vector_path.suffix.lower() in (".geojson", ".json"):
        data = json.loads(vector_path.read_text(encoding="utf-8"))
        crs = data.get("crs", {}).get("properties", {}).get("name")
        return data["features"], crs or "EPSG:4326"
    try:
        import fiona
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("Reading shapefiles / GeoPackages needs 'fiona' (pip install fiona) "
                          "or convert the map to GeoJSON first") from e
    with fiona.open(vector_path) as src:
        crs = src.crs.to_string() if src.crs else None
        return [{"geometry": dict(f["geometry"]), "properties": dict(f["properties"])} for f in src], crs


def rasterize_reference(vector_path, scene_path, out_path, field: str,
                        mapping: Optional[dict] = None) -> Path:
    """Burn geological-map polygons into a label raster aligned with ``scene_path``.

    ``field`` is the attribute holding the unit / lithology. ``mapping`` translates attribute
    values (e.g. "Kohat Limestone", "Qa") to RockMap class ids (1-7); if omitted the field
    must already contain the class ids.
    """
    vector_path = Path(vector_path)
    _, info = read_raster(scene_path, bands=[1])
    feats, src_crs = _read_features(vector_path)
    shapes = []
    unmapped: set = set()
    for f in feats:
        value = f["properties"].get(field)
        cid = mapping.get(str(value)) if mapping else value
        if cid is None or int(cid) not in CLASS_BY_ID:
            unmapped.add(value)
            continue
        geom = f["geometry"]
        if src_crs and info.crs and src_crs != info.crs.to_string():
            geom = transform_geom(src_crs, info.crs, geom)
        shapes.append((geom, int(cid)))
    if not shapes:
        raise ValueError(f"No polygons could be mapped to RockMap classes (field '{field}')")
    labels = rasterize(shapes, out_shape=(info.height, info.width), transform=info.transform,
                       fill=0, dtype="uint8")
    if unmapped:
        print(f"warning: {len(unmapped)} attribute values were not mapped and left unlabelled: "
              f"{sorted(map(str, unmapped))[:10]}")
    return write_raster(out_path, labels[None].astype(np.uint8), info, nodata=0, colormap=class_colormap())


def load_reference_features(vector_path, field: str = "class_id", mapping: Optional[dict] = None) -> list[tuple]:
    """Read a geological map / training polygons as [(geometry in EPSG:4326, class id)]."""
    feats, src_crs = _read_features(Path(vector_path))
    out = []
    for f in feats:
        value = (f.get("properties") or {}).get(field)
        cid = mapping.get(str(value)) if mapping else value
        try:
            cid = int(cid)
        except (TypeError, ValueError):
            continue
        if cid not in CLASS_BY_ID:
            continue
        geom = f["geometry"]
        if src_crs and src_crs not in ("EPSG:4326", "urn:ogc:def:crs:OGC:1.3:CRS84", "OGC:CRS84"):
            geom = transform_geom(src_crs, "EPSG:4326", geom)
        out.append((geom, cid))
    return out
