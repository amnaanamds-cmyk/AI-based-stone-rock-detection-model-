"""XYZ web-map tile rendering (Web Mercator, 256 px) from region mosaics, with a disk cache."""
from __future__ import annotations

import io
import math
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
import rasterio.transform
import rasterio.warp

from .config import CLASS_IDS
from .mapping import colorize

TILE = 256
ORIGIN = 20037508.342789244
_EMPTY: Optional[bytes] = None
_vrt_cache: "OrderedDict[str, tuple]" = OrderedDict()
_vrt_lock = threading.Lock()


def tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    size = 2 * ORIGIN / (2 ** z)
    left = -ORIGIN + x * size
    top = ORIGIN - y * size
    return left, top - size, left + size, top


def empty_png() -> bytes:
    global _EMPTY
    if _EMPTY is None:
        buf = io.BytesIO()
        Image.new("RGBA", (TILE, TILE), (0, 0, 0, 0)).save(buf, "PNG")
        _EMPTY = buf.getvalue()
    return _EMPTY


def _open(path: Path, level: int):
    """Cached dataset handle at an overview level (-1 = full resolution)."""
    key = f"{path}:{path.stat().st_mtime}:{level}"
    with _vrt_lock:
        if key in _vrt_cache:
            _vrt_cache.move_to_end(key)
            return _vrt_cache[key]
        src = rasterio.open(path, overview_level=level) if level >= 0 else rasterio.open(path)
        _vrt_cache[key] = (src, threading.Lock())
        while len(_vrt_cache) > 32:
            _, (s, _l) = _vrt_cache.popitem(last=False)
            s.close()
        return _vrt_cache[key]


def release(prefix: Path | str) -> None:
    """Close cached handles of mosaics under ``prefix`` (Windows cannot replace open files)."""
    prefix = str(prefix)
    with _vrt_lock:
        for key in [k for k in _vrt_cache if k.startswith(prefix)]:
            src, _lock = _vrt_cache.pop(key)
            src.close()


def _pick_level(path: Path, tile_px_m: float) -> int:
    src, _ = _open(path, -1)
    res = abs(src.transform.a)
    factors = src.overviews(1)
    level = -1
    for i, f in enumerate(factors):
        if res * f <= tile_px_m * 1.01:
            level = i
    return level


def render_tile(mosaic: Path, layer: str, z: int, x: int, y: int) -> bytes:
    """PNG bytes of one XYZ tile of a region mosaic (transparent outside data)."""
    categorical = layer in ("lithology", "surface", "hazard", "clusters", "alteration")
    rs = Resampling.nearest if categorical else Resampling.bilinear
    left, bottom, right, top = tile_bounds(z, x, y)
    full, _ = _open(mosaic, -1)
    wb = rasterio.warp.transform_bounds(full.crs, "EPSG:3857", *full.bounds)
    if right <= wb[0] or left >= wb[2] or top <= wb[1] or bottom >= wb[3]:
        return empty_png()
    lat = math.degrees(math.atan(math.sinh(((top + bottom) / 2) / ORIGIN * math.pi)))
    tile_px_m = (right - left) / TILE * math.cos(math.radians(lat))
    src, lock = _open(mosaic, _pick_level(mosaic, tile_px_m))
    dst_transform = rasterio.transform.from_bounds(left, bottom, right, top, TILE, TILE)
    with lock:  # a dataset handle is not thread-safe
        with WarpedVRT(src, crs="EPSG:3857", transform=dst_transform, width=TILE, height=TILE,
                       resampling=rs, src_nodata=0, nodata=0) as vrt:
            data = vrt.read()
    if layer in ("alteration", "hazard", "clusters"):
        rgba = _analytics_rgba(layer, data[0])
    elif categorical:
        rgba = colorize(data[0])
        rgba[data[0] == 0] = 0
        if layer == "surface":
            rgba[data[0] == 1] = 0   # bare rock / soil: transparent so the imagery shows through
    elif layer == "confidence":
        c = data[0].astype(np.float32)
        # low confidence -> red, high -> green
        t = np.clip((c - 30) / 70, 0, 1)
        rgba = np.zeros((TILE, TILE, 4), np.uint8)
        rgba[..., 0] = (220 * (1 - t) + 30 * t).astype(np.uint8)
        rgba[..., 1] = (60 * (1 - t) + 170 * t).astype(np.uint8)
        rgba[..., 2] = 60
        rgba[..., 3] = np.where(c > 0, 230, 0)
    else:
        bands = data[:3] if data.shape[0] >= 3 else np.repeat(data[:1], 3, 0)
        rgba = np.dstack([*bands, np.where(np.any(data > 0, axis=0), 255, 0).astype(np.uint8)])
    if not rgba[..., 3].any():
        return empty_png()
    buf = io.BytesIO()
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(buf, "PNG", compress_level=6)
    return buf.getvalue()


def _lut(colors: dict, alpha: int) -> np.ndarray:
    lut = np.zeros((256, 4), np.uint8)
    for k, c in colors.items():
        c = c.lstrip("#")
        lut[k] = (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha)
    return lut


def _analytics_rgba(layer: str, v: np.ndarray) -> np.ndarray:
    from .analytics import CLUSTER_PALETTE, HAZARD_CLASSES
    if layer == "hazard":
        return _lut({k: c for k, (_n, c) in HAZARD_CLASSES.items()}, 175)[v]
    if layer == "clusters":
        return _lut({i + 1: c for i, c in enumerate((CLUSTER_PALETTE * 16)[:255])}, 210)[v]
    # alteration: score = v - 1; show anomalies >= 40 from yellow to deep red
    score = v.astype(np.float32) - 1
    t = np.clip((score - 40) / 60, 0, 1)
    rgba = np.zeros((*v.shape, 4), np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (230 * (1 - t) + 20 * t).astype(np.uint8)
    rgba[..., 2] = (60 * (1 - t)).astype(np.uint8)
    rgba[..., 3] = np.where(score >= 40, 120 + 135 * t, 0).astype(np.uint8)
    return rgba


def cached_tile(cache_dir: Path, mosaic: Path, layer: str, z: int, x: int, y: int) -> bytes:
    p = cache_dir / layer / str(int(mosaic.stat().st_mtime)) / str(z) / str(x) / f"{y}.png"
    if p.exists():
        return p.read_bytes()
    data = render_tile(mosaic, layer, z, x, y)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    except OSError:
        pass
    return data


def lonlat_zoom_hint(resolution_m: float) -> int:
    """Zoom level whose pixel size is close to the mosaic resolution."""
    return max(0, min(18, int(round(math.log2(2 * ORIGIN / TILE / resolution_m)))))


__all__ = ["render_tile", "cached_tile", "tile_bounds", "CLASS_IDS"]
