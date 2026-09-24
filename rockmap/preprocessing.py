"""Satellite image preprocessing.

* radiometric scaling of digital numbers (DN) to surface reflectance
* simple atmospheric correction (Dark Object Subtraction, DOS1) for top-of-atmosphere data
* cloud / shadow masking (Sentinel-2 SCL, Landsat QA_PIXEL, or the Haze Optimized Transform)
* band stacking of individual band files into one multi-band GeoTIFF
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling

from .config import CANONICAL_BANDS, SENSORS
from .io import GeoInfo, align_to, write_raster

# Sentinel-2 Scene Classification Layer classes treated as invalid:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium, 9 cloud high, 10 cirrus, 11 snow
S2_SCL_INVALID = (0, 1, 3, 8, 9, 10, 11)
# Landsat C2 QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow, 5 snow
LANDSAT_QA_BITS = (0, 1, 2, 3, 4, 5)


def to_reflectance(dn: np.ndarray, sensor: str = "sentinel2") -> np.ndarray:
    """Convert DN values to surface reflectance using the sensor preset scale/offset."""
    preset = SENSORS[sensor]
    refl = dn.astype(np.float32) * preset.scale + preset.offset
    return np.clip(refl, 0.0, 1.5)


def dark_object_subtraction(bands: np.ndarray, percentile: float = 0.5) -> np.ndarray:
    """DOS1 atmospheric correction: subtract each band's dark-object value.

    Path radiance (haze) adds a roughly constant offset to every pixel of a band; the
    darkest pixels in the scene (shadows, deep water) should be ~0 reflectance, so their
    value estimates that offset. Only needed for top-of-atmosphere (L1C / L1) products.
    """
    out = np.empty_like(bands, dtype=np.float32)
    for i, b in enumerate(bands):
        dark = np.nanpercentile(b, percentile) if np.isfinite(b).any() else 0.0
        out[i] = np.clip(b - dark, 0.0, None)
    return out


def hot_cloud_mask(refl: np.ndarray, threshold: float = 0.08) -> np.ndarray:
    """Haze Optimized Transform (Zhang et al., 2002) cloud/haze detection.

    HOT = blue - 0.5 * red. Clear surfaces follow a tight blue-red line; clouds and haze
    are anomalously bright in blue. Returns True where the pixel is cloudy.
    """
    blue, red = refl[0], refl[2]
    swir1 = refl[4]
    hot = blue - 0.5 * red - threshold
    bright = blue > 0.2
    # bright, spectrally flat pixels with high blue are clouds; bright desert/carbonates
    # have strong blue->SWIR increase and are kept.
    flat = swir1 < 1.35 * blue
    return (hot > 0) & bright & flat


def scl_mask(scl: np.ndarray) -> np.ndarray:
    """True where the Sentinel-2 SCL marks cloud, shadow, snow or no data."""
    return np.isin(np.nan_to_num(scl, nan=0).astype(np.int32), S2_SCL_INVALID)


def landsat_qa_mask(qa: np.ndarray) -> np.ndarray:
    """True where Landsat QA_PIXEL flags fill, cloud, cirrus, shadow or snow."""
    qa = np.nan_to_num(qa, nan=1).astype(np.uint32)
    bad = np.zeros(qa.shape, dtype=bool)
    for bit in LANDSAT_QA_BITS:
        bad |= (qa >> bit) & 1 == 1
    return bad


def valid_mask(refl: np.ndarray, cloud: Optional[np.ndarray] = None) -> np.ndarray:
    """True for pixels with finite, non-zero data that are not masked as cloud."""
    ok = np.all(np.isfinite(refl), axis=0) & (np.nansum(refl, axis=0) > 0)
    if cloud is not None:
        ok &= ~cloud
    return ok


def preprocess(raw: np.ndarray, sensor: str = "reflectance", dos: bool = False,
               cloud_layer: Optional[np.ndarray] = None, cloud_layer_type: str = "auto",
               hot_threshold: Optional[float] = 0.08) -> tuple[np.ndarray, np.ndarray]:
    """Full preprocessing chain for a 6-band canonical stack.

    Returns ``(reflectance, valid)`` where ``valid`` is a boolean mask of usable pixels.
    """
    if raw.shape[0] < 6:
        raise ValueError(f"Expected at least 6 bands ({', '.join(CANONICAL_BANDS)}), got {raw.shape[0]}")
    refl = to_reflectance(raw[:6], sensor) if sensor != "reflectance" else raw[:6].astype(np.float32)
    if dos:
        refl = dark_object_subtraction(refl)

    cloud = np.zeros(refl.shape[1:], dtype=bool)
    if cloud_layer is not None:
        kind = cloud_layer_type
        if kind == "auto":
            kind = "scl" if np.nanmax(cloud_layer) <= 11 and sensor.startswith("sentinel2") else (
                "qa" if sensor.startswith("landsat") else "binary")
        if kind == "scl":
            cloud |= scl_mask(cloud_layer)
        elif kind == "qa":
            cloud |= landsat_qa_mask(cloud_layer)
        else:
            cloud |= np.nan_to_num(cloud_layer, nan=0) > 0
    elif hot_threshold is not None:
        cloud |= hot_cloud_mask(np.nan_to_num(refl), hot_threshold)
    return refl, valid_mask(refl, cloud)


# ---------------------------------------------------------------------------
# Band stacking
# ---------------------------------------------------------------------------

def stack_bands(band_paths: Iterable[str | Path], out_path: str | Path,
                resolution: Optional[float] = None, extra_paths: Iterable[str | Path] = ()) -> Path:
    """Stack single-band rasters (e.g. Sentinel-2 JP2 files) into one GeoTIFF.

    All bands are resampled onto the grid of the first band, or onto a grid with the
    requested ``resolution`` (metres) sharing the first band's CRS and extent.
    ``extra_paths`` (e.g. a SCL / QA band) are appended with nearest-neighbour resampling.
    """
    band_paths = [Path(p) for p in band_paths]
    with rasterio.open(band_paths[0]) as src:
        transform, crs, width, height = src.transform, src.crs, src.width, src.height
        if resolution:
            fx, fy = resolution / abs(transform.a), resolution / abs(transform.e)
            width, height = int(round(width / fx)), int(round(height / fy))
            transform = transform @ transform.scale(fx, fy)
    ref = GeoInfo(transform, crs, width, height)
    layers, names = [], []
    for p in band_paths:
        layers.append(align_to(p, ref, Resampling.bilinear))
        names.append(p.stem)
    for p in extra_paths:
        layers.append(align_to(Path(p), ref, Resampling.nearest))
        names.append(Path(p).stem)
    return write_raster(out_path, np.stack(layers).astype(np.float32), ref, descriptions=names)


def find_sentinel2_bands(safe_dir: str | Path, resolution: int = 20) -> tuple[list[Path], Optional[Path]]:
    """Locate the canonical bands (and SCL) inside an unzipped Sentinel-2 L2A .SAFE folder."""
    safe_dir = Path(safe_dir)
    files = list(safe_dir.rglob("*.jp2")) + list(safe_dir.rglob("*.tif"))
    wanted = SENSORS["sentinel2"].band_ids

    def pick(band: str) -> Path:
        cands = [f for f in files if re.search(rf"_{band}(_\d+m)?\.", f.name)]
        if not cands:
            raise FileNotFoundError(f"Band {band} not found in {safe_dir}")
        # prefer the requested resolution, then the finest available
        cands.sort(key=lambda f: (f"_{resolution}m" not in f.name, f.name))
        return cands[0]

    bands = [pick(b) for b in wanted]
    scl = [f for f in files if "_SCL" in f.name]
    return bands, (scl[0] if scl else None)
