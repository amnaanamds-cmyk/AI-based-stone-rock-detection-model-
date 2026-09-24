"""Raster input/output helpers built on rasterio."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window

from .config import CLASS_BY_ID, CLOUD_CLASS, hex_to_rgb


@dataclass
class GeoInfo:
    """Georeferencing of a raster grid."""
    transform: Affine
    crs: Optional[rasterio.crs.CRS]
    width: int
    height: int
    extra: dict = field(default_factory=dict)

    @property
    def pixel_size(self) -> tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)

    @property
    def pixel_area_km2(self) -> float:
        """Pixel area in km^2 (only meaningful for projected CRSs in metres)."""
        px, py = self.pixel_size
        if self.crs is not None and self.crs.is_geographic:
            # approximate: 1 degree ~ 111.32 km
            return (px * 111.32) * (py * 111.32)
        return px * py / 1e6

    def window(self, win: Window) -> "GeoInfo":
        return GeoInfo(rasterio.windows.transform(win, self.transform), self.crs,
                       int(win.width), int(win.height), dict(self.extra))

    def wgs84_bounds(self) -> Optional[tuple[float, float, float, float]]:
        """(west, south, east, north) in EPSG:4326, or None if not georeferenced."""
        if self.crs is None:
            return None
        left, top = self.transform @ (0, 0)
        right, bottom = self.transform @ (self.width, self.height)
        return transform_bounds(self.crs, "EPSG:4326", min(left, right), min(top, bottom),
                                max(left, right), max(top, bottom))


def read_raster(path: str | Path, window: Optional[Window] = None,
                bands: Optional[list[int]] = None) -> tuple[np.ndarray, GeoInfo]:
    """Read a raster as a (bands, rows, cols) float32 array plus its GeoInfo."""
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        data = src.read(idx, window=window).astype(np.float32)
        transform = src.window_transform(window) if window is not None else src.transform
        info = GeoInfo(transform, src.crs, data.shape[2], data.shape[1],
                       {"nodata": src.nodata, "descriptions": list(src.descriptions)})
    if info.extra["nodata"] is not None:
        data[data == info.extra["nodata"]] = np.nan
    return data, info


def raster_shape(path: str | Path) -> tuple[int, int, int]:
    with rasterio.open(path) as src:
        return src.count, src.height, src.width


def write_raster(path: str | Path, data: np.ndarray, info: GeoInfo, nodata=None,
                 descriptions: Optional[list[str]] = None, colormap: Optional[dict] = None) -> Path:
    """Write a (bands, rows, cols) or (rows, cols) array as a compressed GeoTIFF."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data[None]
    profile = dict(driver="GTiff", height=data.shape[1], width=data.shape[2], count=data.shape[0],
                   dtype=data.dtype, transform=info.transform, compress="deflate")
    if info.crs is not None:
        profile["crs"] = info.crs
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
        if descriptions:
            for i, d in enumerate(descriptions, start=1):
                dst.set_band_description(i, d)
        if colormap:
            dst.write_colormap(1, colormap)
    return path


def class_colormap() -> dict[int, tuple[int, int, int, int]]:
    cmap = {0: (0, 0, 0, 0), CLOUD_CLASS: (255, 255, 255, 0)}
    for cid, c in CLASS_BY_ID.items():
        cmap[cid] = (*hex_to_rgb(c.color), 255)
    return cmap


def align_to(src_path: str | Path, ref: GeoInfo, resampling: Resampling = Resampling.bilinear,
             band: int = 1) -> np.ndarray:
    """Reproject/resample one band of ``src_path`` onto the grid described by ``ref``."""
    out = np.full((ref.height, ref.width), np.nan, dtype=np.float32)
    with rasterio.open(src_path) as src:
        if ref.crs is None or src.crs is None:
            # Not georeferenced: fall back to plain resampling of the whole raster.
            data = src.read(band, out_shape=(ref.height, ref.width), resampling=resampling)
            return data.astype(np.float32)
        reproject(source=rasterio.band(src, band), destination=out,
                  src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
                  dst_transform=ref.transform, dst_crs=ref.crs, dst_nodata=np.nan,
                  resampling=resampling)
    return out
