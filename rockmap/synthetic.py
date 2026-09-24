"""Synthetic study-area generator.

Produces a realistic *demo* dataset so the full pipeline (training, classification,
validation, dashboard) can be exercised without downloading real imagery:

* ``scene.tif``      6-band surface reflectance stack (blue, green, red, nir, swir1, swir2)
* ``dem.tif``        elevation model (metres) correlated with rock resistance
* ``reference.tif``  "published" reference geological map (generalised lithology)

Spectral signatures are simplified from typical library spectra (USGS splib07) resampled to
Sentinel-2 bands. Clay/carbonate SWIR absorptions, iron-oxide red slopes, mafic darkness,
vegetation mixing, topographic shading, sensor noise and clouds are all simulated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin
from scipy import ndimage
from scipy.spatial import cKDTree

from .config import CLASS_IDS
from .io import GeoInfo, class_colormap, write_raster

#                     blue  green  red   nir   swir1 swir2
SIGNATURES = {
    1: np.array([0.27, 0.31, 0.35, 0.39, 0.46, 0.36]),   # limestone: bright, CO3 dip in SWIR2
    2: np.array([0.17, 0.22, 0.30, 0.35, 0.42, 0.37]),   # sandstone: iron-stained red slope
    3: np.array([0.14, 0.17, 0.21, 0.25, 0.32, 0.25]),   # shale: clay Al-OH dip in SWIR2
    4: np.array([0.21, 0.24, 0.27, 0.30, 0.35, 0.31]),   # granite: moderately bright, flat
    5: np.array([0.08, 0.09, 0.10, 0.13, 0.16, 0.14]),   # basalt: dark
    6: np.array([0.14, 0.17, 0.20, 0.24, 0.28, 0.25]),   # metamorphic: close to shale, textured
    7: np.array([0.15, 0.19, 0.23, 0.29, 0.36, 0.30]),   # alluvium: mixed detritus
}
VEGETATION = np.array([0.03, 0.07, 0.04, 0.42, 0.21, 0.10])
CLOUD = np.array([0.52, 0.52, 0.53, 0.55, 0.42, 0.33])
# relative resistance to erosion (controls relief)
RESISTANCE = {1: 0.8, 2: 0.6, 3: 0.2, 4: 1.0, 5: 0.9, 6: 0.7, 7: 0.0}
# per-class texture: (amplitude, correlation length in pixels)
TEXTURE = {1: (0.06, 3.0), 2: (0.08, 1.0), 3: (0.12, 0.6), 4: (0.14, 2.5),
           5: (0.12, 1.2), 6: (0.22, 0.8), 7: (0.08, 4.0)}


def _noise(rng: np.random.Generator, shape, sigma: float) -> np.ndarray:
    n = ndimage.gaussian_filter(rng.standard_normal(shape), sigma) if sigma > 0 else rng.standard_normal(shape)
    return (n - n.mean()) / (n.std() + 1e-9)


def _fractal(rng, shape, octaves=(64, 32, 16, 8, 4), decay=0.55) -> np.ndarray:
    out = np.zeros(shape)
    amp = 1.0
    for s in octaves:
        out += amp * _noise(rng, shape, s)
        amp *= decay
    return out / np.abs(out).max()


def _lithology(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Warped Voronoi terrains with folded sedimentary sequences."""
    n_cells = max(12, (h * w) // 9000)
    pts = rng.uniform([0, 0], [h, w], size=(n_cells, 2))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    # domain warping -> irregular, geological-looking contacts
    warp = 0.08 * max(h, w)
    wy = yy + warp * _fractal(rng, (h, w))
    wx = xx + warp * _fractal(rng, (h, w))
    _, cell = cKDTree(pts).query(np.stack([wy.ravel(), wx.ravel()], axis=1))
    cell = cell.reshape(h, w)

    hard_rock = np.array([1, 2, 3, 4, 5, 6])
    cell_class = rng.choice(hard_rock, size=n_cells)
    # guarantee every hard-rock class appears
    cell_class[: len(hard_rock)] = rng.permutation(hard_rock)
    litho = cell_class[cell]

    # some cells become folded sedimentary sequences (sandstone / shale / limestone beds)
    seq = rng.choice(n_cells, size=max(2, n_cells // 5), replace=False)
    theta = rng.uniform(0, np.pi)
    fold = (np.cos(theta) * xx + np.sin(theta) * yy +
            0.06 * max(h, w) * np.sin((np.sin(theta) * xx - np.cos(theta) * yy) / (0.12 * max(h, w))))
    beds = np.array([2, 3, 1, 3])[((fold + 0.05 * max(h, w) * _fractal(rng, (h, w))) // 14).astype(int) % 4]
    in_seq = np.isin(cell, seq)
    litho[in_seq] = beds[in_seq]
    return litho.astype(np.uint8)


def generate_scene(height: int = 512, width: int = 512, seed: int = 42, cloud_cover: float = 0.03,
                   pixel_size: float = 20.0) -> dict:
    """Create a synthetic study area. Returns a dict of arrays + GeoInfo."""
    rng = np.random.default_rng(seed)
    h, w = height, width
    litho = _lithology(rng, h, w)

    # --- DEM: base relief + resistant rocks stand higher ------------------------------
    resist = np.vectorize(RESISTANCE.get)(litho).astype(np.float64)
    dem = 900 + 350 * _fractal(rng, (h, w)) + 500 * ndimage.gaussian_filter(resist, 6)
    # valleys (low ground) are filled with alluvium, flattened
    valley = dem < np.percentile(dem, 14)
    valley = ndimage.binary_opening(valley, iterations=2)
    litho[valley] = 7
    floor = np.percentile(dem, 10)
    dem[valley] = floor + 0.25 * (dem[valley] - floor) + 3 * rng.standard_normal(valley.sum())
    dem = ndimage.gaussian_filter(dem, 1.0)

    # --- reflectance -------------------------------------------------------------------
    refl = np.zeros((6, h, w))
    for cid, sig in SIGNATURES.items():
        m = litho == cid
        refl[:, m] = sig[:, None]
    # regional brightness variation (weathering, varnish) and class-specific texture
    refl *= 1 + 0.16 * _fractal(rng, (h, w), octaves=(48, 24))[None]
    # band-wise weathering tint (desert varnish, iron coatings) blurs spectral separability
    refl *= 1 + 0.06 * np.stack([_fractal(rng, (h, w), octaves=(40, 20)) for _ in range(6)])
    for cid, (amp, corr) in TEXTURE.items():
        m = litho == cid
        refl[:, m] *= (1 + amp * _noise(rng, (h, w), corr))[m][None]
    # sub-pixel mixing along contacts (sensor PSF)
    refl = ndimage.gaussian_filter(refl, sigma=(0, 0.6, 0.6))

    # vegetation fraction: denser on alluvium and shale
    veg = np.clip(0.22 * _fractal(rng, (h, w), octaves=(24, 12, 6)) + 0.08, 0, 0.5)
    veg[litho == 7] += 0.20
    veg[litho == 3] += 0.06
    veg = np.clip(veg, 0, 0.7)
    refl = (1 - veg)[None] * refl + veg[None] * VEGETATION[:, None, None]

    # topographic illumination (Lambertian, sun from NW at 45 deg)
    dy, dx = np.gradient(dem, pixel_size)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(45)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    refl *= (0.55 + 0.45 * np.clip(shade, 0, 1) / np.sin(alt))[None]

    # clouds
    cloud_mask = np.zeros((h, w), dtype=bool)
    if cloud_cover > 0:
        c = _fractal(rng, (h, w), octaves=(20, 10, 5))
        thick = np.clip((c - np.quantile(c, 1 - cloud_cover)) * 8, 0, 1)
        cloud_mask = thick > 0.35
        refl = (1 - thick)[None] * refl + thick[None] * CLOUD[:, None, None]

    refl += 0.012 * rng.standard_normal(refl.shape)          # sensor noise
    refl = np.clip(refl, 0.001, 1.0).astype(np.float32)

    # "published" reference map: generalised (majority filtered) true lithology
    onehot = np.stack([ndimage.uniform_filter((litho == c).astype(np.float32), 3) for c in CLASS_IDS])
    reference = np.asarray(CLASS_IDS, dtype=np.uint8)[onehot.argmax(0)]

    # UTM zone 42N, around Charsadda / Peshawar basin (Khyber Pakhtunkhwa)
    info = GeoInfo(from_origin(620000, 3810000, pixel_size, pixel_size), CRS.from_epsg(32642), w, h)
    return {"reflectance": refl, "dem": dem.astype(np.float32), "reference": reference,
            "truth": litho, "cloud": cloud_mask, "info": info}


def write_scene(out_dir: str | Path, height: int = 512, width: int = 512, seed: int = 42,
                cloud_cover: float = 0.03) -> dict[str, Path]:
    """Generate a synthetic scene and write it as GeoTIFFs into ``out_dir``."""
    out_dir = Path(out_dir)
    s = generate_scene(height, width, seed, cloud_cover)
    info = s["info"]
    paths = {
        "scene": write_raster(out_dir / "scene.tif", s["reflectance"], info,
                              descriptions=["blue", "green", "red", "nir", "swir1", "swir2"]),
        "dem": write_raster(out_dir / "dem.tif", s["dem"][None], info, descriptions=["elevation"]),
        "reference": write_raster(out_dir / "reference.tif", s["reference"][None], info, nodata=0,
                                  colormap=class_colormap()),
    }
    return paths
