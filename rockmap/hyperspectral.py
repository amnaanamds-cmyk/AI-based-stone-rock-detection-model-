"""Hyperspectral mineral mapping (EnMAP, PRISMA, AVIRIS-NG, GF-5 AHSI...).

Sentinel-2 has two SWIR bands, so it sees "clay" but cannot tell kaolinite from alunite or sericite -
exactly the minerals that zone a porphyry-copper system (advanced argillic -> argillic -> phyllic ->
propylitic). Hyperspectral sensors sample the 2000-2450 nm window every ~10 nm, which resolves the
diagnostic absorption features.

Two methods are provided:

1. **Diagnostic absorption features** (no library needed). For each mineral the depth of its
   characteristic absorption is measured against a straight-line continuum between two shoulders
   (continuum-removed band depth, the basis of the USGS Tetracorder approach). Secondary features
   separate look-alikes, e.g. kaolinite's 2165/2205 nm doublet versus sericite's single 2200 nm band.

   ==============  ==================================  ============================================
   mineral         diagnostic features (nm)            alteration / meaning
   ==============  ==================================  ============================================
   alunite         1762, 2165 (> 2205)                 advanced argillic (porphyry lithocap, epithermal)
   kaolinite       2165 + 2205 doublet                 argillic
   sericite        2200 (single), 2350                 phyllic (muscovite / illite) - porphyry core
   chlorite        2250 + 2335                         propylitic (also epidote)
   calcite         2340 (weak 2200)                    carbonate, marble, skarn host
   jarosite        2265                                acid sulfate weathering of sulfides
   hematite        ~870 (Fe3+ crystal field)           iron oxide, gossan
   goethite        ~920                                iron oxide, gossan
   ==============  ==================================  ============================================

2. **Spectral Angle Mapper** against a user spectral library (CSV: ``wavelength_nm`` column plus one
   column per reference spectrum, e.g. exported from the USGS Spectral Library splib07). Stibnite and
   quartz have no SWIR features and cannot be mapped this way.

The cube is processed block by block and only the bands that are needed are read, so full EnMAP
scenes (~1200 x 1200 x 224) fit in a laptop's memory.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.windows import Window

EPS = 1e-6


@dataclass(frozen=True)
class Feature:
    left: float
    center: float
    right: float


@dataclass(frozen=True)
class Mineral:
    key: str
    name: str
    group: str
    color: str
    main: Feature
    secondary: tuple = ()          # (Feature, min_ratio) pairs that must also be present
    absent: tuple = ()             # (Feature, max_ratio) pairs that must be weak relative to ``main``


MINERALS: tuple[Mineral, ...] = (
    Mineral("alunite", "Alunite", "advanced argillic", "#d73027", Feature(2120, 2165, 2195),
            secondary=((Feature(1720, 1762, 1800), 0.25),), absent=((Feature(2185, 2205, 2235), 0.9),)),
    Mineral("kaolinite", "Kaolinite", "argillic", "#fc8d59", Feature(2120, 2205, 2245),
            secondary=((Feature(2120, 2165, 2185), 0.35),)),
    Mineral("sericite", "Sericite / muscovite / illite", "phyllic", "#fee090", Feature(2120, 2200, 2260),
            secondary=((Feature(2300, 2350, 2400), 0.2),), absent=((Feature(2120, 2165, 2185), 0.35),)),
    Mineral("chlorite", "Chlorite / epidote", "propylitic", "#1a9850", Feature(2220, 2250, 2280),
            secondary=((Feature(2280, 2335, 2380), 0.5),)),
    Mineral("calcite", "Calcite / carbonate", "carbonate", "#4575b4", Feature(2280, 2340, 2400),
            absent=((Feature(2120, 2200, 2260), 0.6), (Feature(2220, 2250, 2280), 0.5))),
    Mineral("jarosite", "Jarosite", "acid sulfate", "#b8860b", Feature(2230, 2265, 2290)),
    Mineral("hematite", "Hematite", "iron oxide", "#a50026", Feature(750, 870, 1000)),
    Mineral("goethite", "Goethite", "iron oxide", "#8c510a", Feature(750, 925, 1150)),
)
MINERAL_BY_KEY = {m.key: m for m in MINERALS}
MIN_DEPTH = 0.03                   # continuum-removed band depth below which nothing is mapped
CLAY_KEYS = ("alunite", "kaolinite", "sericite")
IRON_KEYS = ("hematite", "goethite")


# ----------------------------------------------------------------------------- wavelengths
def _to_nm(values: Sequence[float]) -> np.ndarray:
    w = np.asarray(values, np.float64)
    if np.nanmax(w) < 30:          # micrometres
        w = w * 1000.0
    return w


def parse_wavelengths(text: str) -> np.ndarray:
    """Wavelengths from an ENVI .hdr (``wavelength = {...}``), a CSV/TXT list, or any text with numbers."""
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.I)
    body = m.group(1) if m else text
    nums = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", body)]
    if len(nums) < 10:
        raise ValueError("could not find at least 10 wavelengths")
    return _to_nm(nums)


def cube_wavelengths(path: str | Path, wavelengths: Optional[Sequence[float]] = None) -> np.ndarray:
    """Band centre wavelengths (nm): given explicitly, from band descriptions / tags, or a sidecar .hdr."""
    if wavelengths is not None:
        return _to_nm(wavelengths)
    path = Path(path)
    with rasterio.open(path) as src:
        vals = []
        for i in range(1, src.count + 1):
            tags = src.tags(i)
            txt = tags.get("wavelength") or tags.get("WAVELENGTH") or src.descriptions[i - 1] or ""
            found = re.findall(r"\d+(?:\.\d+)?", str(txt))
            vals.append(float(found[0]) if found else np.nan)
        if np.isfinite(vals).all():
            return _to_nm(vals)
        for key in ("wavelength", "WAVELENGTH"):
            if src.tags().get(key):
                return parse_wavelengths(src.tags()[key])
    for side in (path.with_suffix(".hdr"), Path(str(path) + ".hdr")):
        if side.exists():
            return parse_wavelengths(side.read_text(encoding="utf-8", errors="ignore"))
    raise ValueError("band wavelengths not found: give them in band descriptions, a .hdr file or explicitly")


# ----------------------------------------------------------------------------- band depth
def _nearest(wl: np.ndarray, target: float, tol: float = 25.0) -> Optional[int]:
    i = int(np.nanargmin(np.abs(wl - target)))
    return i if abs(wl[i] - target) <= tol else None


def feature_bands(wl: np.ndarray, f: Feature) -> Optional[tuple[int, int, int]]:
    idx = (_nearest(wl, f.left), _nearest(wl, f.center, 15.0), _nearest(wl, f.right))
    return None if None in idx or len(set(idx)) < 3 else idx   # type: ignore[return-value]


def band_depth(left: np.ndarray, center: np.ndarray, right: np.ndarray,
               wl: tuple[float, float, float]) -> np.ndarray:
    """1 - R_center / continuum, the continuum being the straight line between the two shoulders."""
    t = (wl[1] - wl[0]) / (wl[2] - wl[0])
    cont = left * (1 - t) + right * t
    with np.errstate(divide="ignore", invalid="ignore"):
        d = 1.0 - center / cont
    return np.where(np.isfinite(d) & (cont > 0.02), d, 0.0).astype(np.float32)


def needed_bands(wl: np.ndarray) -> list[int]:
    out = set()
    for m in MINERALS:
        for f in [m.main] + [s[0] for s in m.secondary] + [a[0] for a in m.absent]:
            idx = feature_bands(wl, f)
            if idx:
                out.update(idx)
    return sorted(out)


def mineral_depths(bands: dict[int, np.ndarray], wl: np.ndarray) -> dict[str, np.ndarray]:
    """Per-mineral depth maps (0 where secondary / absent checks fail) from a {band index: array} dict."""
    def depth(f):
        idx = feature_bands(wl, f)
        if idx is None:
            return None
        return band_depth(bands[idx[0]], bands[idx[1]], bands[idx[2]], tuple(wl[list(idx)]))

    out = {}
    for m in MINERALS:
        d = depth(m.main)
        if d is None:
            continue
        ok = d >= MIN_DEPTH
        for f, ratio in m.secondary:
            s = depth(f)
            if s is not None:
                ok &= s >= ratio * d
        for f, ratio in m.absent:
            a = depth(f)
            if a is not None:
                ok &= a <= ratio * d
        out[m.key] = np.where(ok, d, 0).astype(np.float32)
    return out


def classify(depths: dict[str, np.ndarray]) -> np.ndarray:
    """Dominant SWIR mineral (1-based index into MINERALS) where any depth >= MIN_DEPTH; iron oxides are
    mapped separately because they coexist with clays (VNIR vs SWIR features)."""
    swir = [m.key for m in MINERALS if m.key not in IRON_KEYS and m.key in depths]
    shape = next(iter(depths.values())).shape
    if not swir:
        return np.zeros(shape, np.uint8)
    stack = np.stack([depths[k] for k in swir])
    best = np.argmax(stack, axis=0)
    code = np.asarray([[m.key for m in MINERALS].index(k) + 1 for k in swir], np.uint8)[best]
    return np.where(stack.max(axis=0) >= MIN_DEPTH, code, 0).astype(np.uint8)


# ----------------------------------------------------------------------------- spectral library / SAM
def parse_library(data: bytes) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Spectral library CSV: ``wavelength`` (nm or um) column + one reflectance column per mineral."""
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    head = [h.strip() for h in rows[0]]
    body = np.asarray([[float(v) if v.strip() else np.nan for v in r] for r in rows[1:] if r], np.float64)
    wl = _to_nm(body[:, 0])
    return wl, {h: body[:, i] for i, h in enumerate(head) if i > 0}


def sam(cube: np.ndarray, cube_wl: np.ndarray, lib_wl: np.ndarray, library: dict[str, np.ndarray],
        window: tuple[float, float] = (2050.0, 2400.0), max_angle: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """Spectral Angle Mapper on continuum-removed absorption depths in the SWIR window.

    Comparing raw spectra lets overall slope and brightness dominate (a featureless rock would
    "match" any mineral), so both cube and library are converted to band depths below a straight
    continuum across the window first. Returns (class 1..n or 0, angle in radians).
    """
    sel = (cube_wl >= window[0]) & (cube_wl <= window[1])
    w = cube_wl[sel]
    t = (w - w[0]) / (w[-1] - w[0] + EPS)

    def depth(s):          # s: (bands, n)
        cont = s[:1] * (1 - t[:, None]) + s[-1:] * t[:, None]
        cont = np.maximum(cont, s)             # continuum never below the spectrum (hull-like)
        return 1.0 - s / (cont + EPS)

    x = depth(cube[sel].reshape(sel.sum(), -1).astype(np.float64))
    refs = []
    for v in library.values():
        ok = np.isfinite(v)
        refs.append(np.interp(w, lib_wl[ok], v[ok]))
    r = depth(np.asarray(refs).T).T
    xn = x / (np.linalg.norm(x, axis=0) + EPS)
    rn = r / (np.linalg.norm(r, axis=1, keepdims=True) + EPS)
    ang = np.arccos(np.clip(rn @ xn, -1, 1))
    best = ang.argmin(axis=0)
    bang = ang.min(axis=0)
    has_features = x.max(axis=0) >= MIN_DEPTH
    cls = np.where((bang <= max_angle) & np.isfinite(x).all(axis=0) & has_features, best + 1, 0)
    shape = cube.shape[1:]
    return cls.reshape(shape).astype(np.uint8), bang.reshape(shape).astype(np.float32)


# ----------------------------------------------------------------------------- whole scene
def _scale(sample: np.ndarray) -> float:
    """Reflectance scale factor: EnMAP / PRISMA products are often stored as integers x 10000."""
    v = sample[np.isfinite(sample) & (sample > 0)]
    return 1e-4 if v.size and np.percentile(v, 50) > 2.0 else 1.0


def map_scene(path: str | Path, out_path: str | Path, wavelengths: Optional[Sequence[float]] = None,
              library: Optional[tuple[np.ndarray, dict]] = None, block: int = 512) -> dict:
    """Mineral maps of one hyperspectral scene, written in the scene's own grid.

    Output bands: 1 dominant SWIR mineral code + 1 (1 = measured, no SWIR mineral; 0 = no data), 2..n+1 band depth x 1000 per mineral (see ``MINERALS``),
    and - with a library - one more band with the SAM class. Returns a summary (pixel counts).
    """
    wl = cube_wavelengths(path, wavelengths)
    need = needed_bands(wl)
    if not need:
        raise ValueError("the scene has no bands in the diagnostic 750-2400 nm features")
    counts = {m.key: 0 for m in MINERALS}
    with rasterio.open(path) as src:
        if src.count != len(wl):
            raise ValueError(f"{src.count} bands but {len(wl)} wavelengths")
        nodata = src.nodata
        scale = _scale(src.read(need[len(need) // 2] + 1, out_shape=(64, 64)).astype(np.float32))
        sam_bands = []
        if library:
            sam_bands = [i for i in range(len(wl)) if 2000 <= wl[i] <= 2450]
        prof = dict(driver="GTiff", width=src.width, height=src.height, crs=src.crs, transform=src.transform,
                    count=1 + len(MINERALS) + (1 if library else 0), dtype="uint16", nodata=0,
                    compress="deflate", tiled=True, blockxsize=256, blockysize=256)
        with rasterio.open(out_path, "w", **prof) as dst:
            for row in range(0, src.height, block):
                for col in range(0, src.width, block):
                    win = Window(col, row, min(block, src.width - col), min(block, src.height - row))
                    data = src.read([i + 1 for i in need], window=win).astype(np.float32)
                    valid = np.all(np.isfinite(data), axis=0) & (data.sum(axis=0) > 0)
                    if nodata is not None:
                        valid &= np.all(data != nodata, axis=0)
                    bands = {b: data[j] * scale for j, b in enumerate(need)}
                    depths = mineral_depths(bands, wl)
                    cls = np.where(valid, classify(depths), 0)
                    out = [np.where(valid, cls.astype(np.uint16) + 1, 0)]
                    for m in MINERALS:
                        d = depths.get(m.key, np.zeros(cls.shape, np.float32))
                        d = np.where(valid, d, 0)
                        counts[m.key] += int((d >= MIN_DEPTH).sum())
                        out.append((np.clip(d, 0, 1) * 1000).astype(np.uint16))
                    if library:
                        cube = src.read([i + 1 for i in sam_bands], window=win).astype(np.float32) * scale
                        sc, _ang = sam(cube, wl[sam_bands], library[0], library[1])
                        out.append(np.where(valid, sc, 0).astype(np.uint16))
                    dst.write(np.stack(out), window=win)
            dst.update_tags(minerals=",".join(m.key for m in MINERALS),
                            library=",".join(library[1]) if library else "")
    return {"wavelengths": int(len(wl)), "range_nm": [round(float(wl.min())), round(float(wl.max()))],
            "pixels": counts, "library": list(library[1]) if library else []}
