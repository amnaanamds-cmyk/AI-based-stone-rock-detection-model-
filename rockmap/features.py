"""Feature extraction: spectral band ratios / indices and DEM-derived terrain features."""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import ndimage

EPS = 1e-6

BAND_FEATURES = ["blue", "green", "red", "nir", "swir1", "swir2"]
INDEX_FEATURES = [
    "ndvi",          # (NIR - Red) / (NIR + Red)          vegetation cover
    "clay_ratio",    # SWIR1 / SWIR2                      Al-OH / Mg-OH / CO3 absorption (clays, carbonates)
    "iron_oxide",    # Red / Blue                         ferric iron (hematite, goethite)
    "ferrous",       # SWIR1 / NIR                        ferrous iron (mafic minerals)
    "ferric_ratio",  # Red / Green                        iron staining
    "brightness",    # mean reflectance                   albedo
    "ndbi_rock",     # (SWIR1 - NIR) / (SWIR1 + NIR)       bare rock / soil exposure
]
TERRAIN_FEATURES = ["elevation", "slope", "aspect_sin", "aspect_cos", "hillshade", "roughness"]


def spectral_indices(refl: np.ndarray) -> np.ndarray:
    """Compute band ratios / indices from a canonical 6-band reflectance stack."""
    blue, green, red, nir, swir1, swir2 = refl[:6]
    feats = [
        (nir - red) / (nir + red + EPS),
        swir1 / (swir2 + EPS),
        red / (blue + EPS),
        swir1 / (nir + EPS),
        red / (green + EPS),
        refl[:6].mean(axis=0),
        (swir1 - nir) / (swir1 + nir + EPS),
    ]
    out = np.stack(feats).astype(np.float32)
    # ratios can explode over deep shadow / no-data; keep them in a sane range
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)


def terrain_features(dem: np.ndarray, pixel_size: float = 20.0,
                     sun_azimuth: float = 315.0, sun_elevation: float = 45.0) -> np.ndarray:
    """Slope, aspect, hillshade and roughness from an elevation model (metres)."""
    dem = dem.astype(np.float32)
    if not np.isfinite(dem).all():
        fill = np.nanmean(dem) if np.isfinite(dem).any() else 0.0
        dem = np.where(np.isfinite(dem), dem, fill)
    dz_dy, dz_dx = np.gradient(dem, pixel_size)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))                      # radians
    aspect = np.arctan2(-dz_dx, dz_dy)                             # radians, 0 = north-facing
    az, alt = np.radians(sun_azimuth), np.radians(sun_elevation)
    hill = (np.sin(alt) * np.cos(slope) +
            np.cos(alt) * np.sin(slope) * np.cos(az - aspect))
    rough = dem - ndimage.uniform_filter(dem, size=5)             # local relief (TPI-like)
    elev = (dem - np.mean(dem)) / (np.std(dem) + EPS)             # scene-normalised elevation
    return np.stack([
        elev,
        np.degrees(slope) / 90.0,
        np.sin(aspect),
        np.cos(aspect),
        np.clip(hill, 0, 1),
        np.clip(rough / 50.0, -5, 5),
    ]).astype(np.float32)


def build_features(refl: np.ndarray, dem: Optional[np.ndarray] = None,
                   pixel_size: float = 20.0) -> tuple[np.ndarray, list[str]]:
    """Stack bands + indices (+ terrain) into a (features, rows, cols) array."""
    parts = [np.nan_to_num(refl[:6]).astype(np.float32), spectral_indices(np.nan_to_num(refl))]
    names = BAND_FEATURES + INDEX_FEATURES
    if dem is not None:
        parts.append(terrain_features(dem, pixel_size))
        names = names + TERRAIN_FEATURES
    return np.concatenate(parts, axis=0), list(names)


class Normalizer:
    """Per-feature standardisation (z-score) fitted on training pixels."""

    def __init__(self, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None):
        self.mean = mean
        self.std = std

    def fit(self, samples: np.ndarray) -> "Normalizer":
        """``samples`` shape: (n_samples, n_features)."""
        self.mean = samples.mean(axis=0).astype(np.float32)
        self.std = (samples.std(axis=0) + 1e-6).astype(np.float32)
        return self

    def transform_pixels(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def transform_image(self, img: np.ndarray) -> np.ndarray:
        return ((img - self.mean[:, None, None]) / self.std[:, None, None]).astype(np.float32)

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(np.asarray(d["mean"], np.float32), np.asarray(d["std"], np.float32))
