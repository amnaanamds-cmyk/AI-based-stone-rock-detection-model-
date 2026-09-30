"""Structural lineaments and multi-commodity mineral prospectivity.

Commercial exploration programmes rarely ask "where is clay alteration?"; they ask "where should we
look for copper, iron or vein-hosted antimony / gold?". Each commodity has its own evidence, so each
gets its own knowledge-driven (fuzzy) model:

==========  ==================================  =============================================================
model       commodities                         evidence
==========  ==================================  =============================================================
``iron``    iron ore (haematite, magnetite      ferric iron oxide (red/blue), gossan (SWIR1/red), structures
            gossans)
``copper``  porphyry / vein copper (+Au, Mo)    clay / sericite (Al-OH, SWIR1/SWIR2) with iron-oxide cap,
                                                on fractured ground (lineament density)
``vein``    quartz-vein antimony, gold,         structural control (lineament density and proximity) and pale,
            quartz                              iron-poor siliceous rock; the ASTER thermal Quartz Index
                                                replaces the Sentinel-2 proxy when ASTER is imported
==========  ==================================  =============================================================

Honest limits (also printed in the report):

* Sentinel-2 has no thermal bands: quartz itself is only detectable in the thermal infrared
  (ASTER bands 10-14). Without ASTER the ``vein`` model is a *structural* model.
* Antimony (stibnite) has no diagnostic spectral signature at any resolution; it is targeted
  through its quartz-vein host and structural setting only.
* Clay minerals (kaolinite, alunite, sericite) and sulfates cannot be told apart with two SWIR
  bands; hyperspectral data (EnMAP, PRISMA) or ASTER SWIR is needed for that.

Lineaments are extracted automatically from the DEM: edges of hillshades lit from four directions,
kept only where they continue in a straight line for several hundred metres (fault- and
fracture-controlled valleys, scarps and ridges). They are *candidate* structures for a geologist to
interpret, delivered as GeoJSON lines, a density raster and a rose diagram of strike directions.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage, signal

from .analytics import alteration_indices
from .preprocessing import illumination

EPS = 1e-6

COMMODITY_TYPES = {
    "iron": "iron", "iron ore": "iron", "haematite": "iron", "hematite": "iron", "magnetite": "iron",
    "copper": "copper", "cu": "copper", "porphyry": "copper", "molybdenum": "copper",
    "antimony": "vein", "stibnite": "vein", "sb": "vein", "gold": "vein", "au": "vein",
    "quartz": "vein", "quartz vein": "vein", "silica": "vein",
}
COMMODITY_NAMES = ("iron", "copper", "antimony", "gold", "quartz")


@dataclass(frozen=True)
class MineralModel:
    key: str
    name: str
    commodities: str
    color: str
    setting: str
    weights: dict


MINERAL_MODELS: tuple[MineralModel, ...] = (
    MineralModel("iron", "Iron oxide / iron ore", "iron ore (haematite, magnetite), gossans", "#b2182b",
                 "Strong ferric iron oxide (red/blue) and gossan (SWIR1/red) response, favoured on fractured ground.",
                 {"iron_oxide": 0.55, "gossan": 0.30, "lineament": 0.15}),
    MineralModel("copper", "Copper alteration (porphyry / vein)", "copper (+ gold, molybdenum)", "#1b7837",
                 "Clay / sericite alteration (Al-OH, SWIR1/SWIR2) with an iron-oxide cap on structurally "
                 "prepared (high lineament density) ground.",
                 {"clay": 0.35, "iron_oxide": 0.25, "lineament": 0.25, "gossan": 0.15}),
    MineralModel("vein", "Quartz veins (antimony, gold)", "antimony (stibnite), gold, quartz", "#e08214",
                 "Structural model: dense and nearby lineaments (faults, shear zones) in pale, iron-poor "
                 "siliceous rock. With ASTER imported, the thermal Quartz Index carries most of the weight.",
                 {"lineament": 0.30, "proximity": 0.20, "silica": 0.50}),
)
MODEL_BY_KEY = {m.key: m for m in MINERAL_MODELS}
VEIN_WEIGHTS_WITH_ASTER = {"lineament": 0.25, "proximity": 0.20, "silica": 0.10, "quartz_index": 0.45}
EVIDENCE = ("clay", "iron_oxide", "gossan", "brightness", "lineament", "quartz_index", "hyper_clay", "hyper_iron")
# where hyperspectral mineral maps exist, they replace most of the broad Sentinel-2 ratios
HYPER_WEIGHTS = {
    "iron": {"hyper_iron": 0.40, "iron_oxide": 0.25, "gossan": 0.20, "lineament": 0.15},
    "copper": {"hyper_clay": 0.40, "clay": 0.10, "iron_oxide": 0.15, "lineament": 0.25, "gossan": 0.10},
}
FEATURE_KEYS = ("clay", "iron_oxide", "gossan", "silica", "lineament", "proximity")
TARGET_THRESHOLD = 75
TARGET_PERCENTILE = 99.0
MAX_TARGETS_PER_MODEL = 50


# ----------------------------------------------------------------------------- lineaments
def _line_kernel(length: int, angle_deg: float) -> np.ndarray:
    """1-pixel-wide straight line of ``length`` pixels at map azimuth ``angle_deg`` (0 = north)."""
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2
    t = np.linspace(-c, c, length * 2)
    a = np.radians(angle_deg)
    rows = np.round(c - t * np.cos(a)).astype(int)
    cols = np.round(c + t * np.sin(a)).astype(int)
    k[rows, cols] = 1
    return k / k.sum()


def lineaments(dem: np.ndarray, pixel_size: float, length_m: float = 600.0, n_dir: int = 12,
               edge_percentile: float = 85.0, straightness: float = 0.7,
               valid: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Candidate structural lineaments from a DEM.

    Returns ``(mask, strike)``: boolean lineament pixels and their strike in degrees (0-180,
    clockwise from north; NaN elsewhere). Edges are detected on hillshades lit from four
    directions (a single sun direction hides structures parallel to it); a pixel is a lineament
    when at least ``straightness`` of a ``length_m`` long line through it is also edge.
    """
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0.0).astype(np.float64)
    ok = np.isfinite(dem) if valid is None else valid & np.isfinite(dem)
    edges = np.zeros(d.shape, bool)
    for az in (0, 45, 90, 135):
        hs = illumination(d, pixel_size, az, 30.0)
        mag = np.hypot(ndimage.sobel(hs, 0), ndimage.sobel(hs, 1))
        if ok.any():
            edges |= mag >= np.percentile(mag[ok], edge_percentile)
    edges &= ok
    length = max(9, int(round(length_m / pixel_size)) | 1)
    best = np.zeros(d.shape, np.float32)
    strike = np.full(d.shape, np.nan, np.float32)
    e = edges.astype(np.float32)
    for i in range(n_dir):
        ang = 180.0 * i / n_dir
        resp = signal.fftconvolve(e, _line_kernel(length, ang), mode="same")
        better = resp > best
        best[better] = resp[better]
        strike[better] = ang
    mask = edges & (best >= straightness)
    strike[~mask] = np.nan
    return mask, strike


def lineament_density(mask: np.ndarray, pixel_size: float, radius_m: float = 1000.0) -> np.ndarray:
    """Lineament length per area (km / km2) in a moving window of about ``2 * radius_m``."""
    win = max(3, int(round(2 * radius_m / pixel_size)) | 1)
    frac = ndimage.uniform_filter(mask.astype(np.float32), win)
    return (frac * 1000.0 / pixel_size).astype(np.float32)


def lineament_segments(mask: np.ndarray, strike: np.ndarray, transform, pixel_size: float,
                       min_length_m: float = 400.0, n_dir: int = 12) -> list[dict]:
    """Vectorise lineament pixels into straight segments (one per connected run of equal strike)."""
    out = []
    step = 180.0 / n_dir
    bins = np.where(mask, np.round(np.nan_to_num(strike) / step).astype(int) % n_dir, -1)
    for b in range(n_dir):
        lab, n = ndimage.label(bins == b, structure=np.ones((3, 3)))
        if not n:
            continue
        ang = np.radians(b * step)
        dc, dr = np.sin(ang), -np.cos(ang)            # one step along strike in (col, row) pixels
        for sl, idx in zip(ndimage.find_objects(lab), range(1, n + 1)):
            rr, cc = np.nonzero(lab[sl] == idx)
            if len(rr) * pixel_size < min_length_m:
                continue
            rr = rr + sl[0].start
            cc = cc + sl[1].start
            r0, c0 = rr.mean(), cc.mean()
            t = (cc - c0) * dc + (rr - r0) * dr
            p1 = (c0 + t.min() * dc, r0 + t.min() * dr)
            p2 = (c0 + t.max() * dc, r0 + t.max() * dr)
            length = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]) * pixel_size)
            if length < min_length_m:
                continue
            (x1, y1), (x2, y2) = transform * (p1[0] + 0.5, p1[1] + 0.5), transform * (p2[0] + 0.5, p2[1] + 0.5)
            out.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "strike": round(b * step, 1),
                        "length_m": round(length, 0)})
    return out


def rose(segments: list[dict], n_bins: int = 12) -> list[dict]:
    """Total lineament length (km) per strike sector - the data behind a rose diagram."""
    step = 180.0 / n_bins
    tot = np.zeros(n_bins)
    for s in segments:
        tot[int(s["strike"] // step) % n_bins] += s["length_m"] / 1000.0
    return [{"from": round(i * step, 1), "to": round((i + 1) * step, 1), "km": round(float(v), 2)}
            for i, v in enumerate(tot)]


# ----------------------------------------------------------------------------- evidence & models
@dataclass
class MineralStats:
    median: dict
    scale: dict

    def z(self, key: str, v: np.ndarray) -> np.ndarray:
        return (v - self.median[key]) / self.scale[key]

    def to_dict(self) -> dict:
        return {"median": self.median, "scale": self.scale}

    @classmethod
    def from_dict(cls, d: dict) -> "MineralStats":
        return cls(d["median"], d["scale"])

    @classmethod
    def fit(cls, samples: dict[str, np.ndarray]) -> "MineralStats":
        med, sc = {}, {}
        for k, v in samples.items():
            v = v[np.isfinite(v)]
            m = float(np.median(v)) if v.size else 0.0
            mad = float(np.median(np.abs(v - m))) * 1.4826 if v.size else 1.0
            med[k], sc[k] = m, max(mad, 1e-4)
        return cls(med, sc)


def quartz_index(emissivity: np.ndarray) -> np.ndarray:
    """ASTER thermal Quartz Index (Rockwell & Hofstra 2008): QI = e11^2 / (e10 * e12).

    ``emissivity`` holds ASTER bands 10, 11, 12 (and optionally 13, 14) in that order.
    Quartz has a strong reststrahlen feature in band 12 relative to 11, so QI rises with quartz.
    """
    e10, e11, e12 = (np.asarray(b, np.float64) for b in emissivity[:3])
    with np.errstate(divide="ignore", invalid="ignore"):
        qi = e11 * e11 / (e10 * e12)
    return np.where(np.isfinite(qi) & (e10 > 0) & (e12 > 0), qi, np.nan).astype(np.float32)


def raw_evidence(refl: np.ndarray, lin_density: np.ndarray, qi: Optional[np.ndarray] = None) -> dict:
    idx = alteration_indices(refl)
    out = {"clay": idx["clay"], "iron_oxide": idx["iron_oxide"], "gossan": idx["gossan"],
           "brightness": np.nan_to_num(refl[:6]).mean(axis=0), "lineament": lin_density}
    out["quartz_index"] = qi if qi is not None else np.full(lin_density.shape, np.nan, np.float32)
    return out


def _pos(z: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(z) / 3.0, 0, 1)


def evidence_layers(raw: dict, usable: np.ndarray, stats: MineralStats, lin_mask: np.ndarray,
                    pixel_size: float) -> dict[str, np.ndarray]:
    """Fuzzy memberships (0..1) for every evidence layer."""
    z = {k: stats.z(k, raw[k]) for k in EVIDENCE if k in stats.median}
    iron = _pos(z["iron_oxide"])
    dist = ndimage.distance_transform_edt(~lin_mask) * pixel_size if lin_mask.any() else \
        np.full(lin_mask.shape, 1e6)
    ev = {
        "clay": _pos(z["clay"]),
        "iron_oxide": iron,
        "gossan": _pos(z["gossan"]),
        # pale, iron-poor rock: quartz veins, silicified zones and leucogranite are bright and iron-free
        "silica": _pos(z["brightness"]) * (1 - iron),
        "lineament": _pos(z["lineament"]),
        "proximity": np.exp(-dist / 100.0),
    }
    if "quartz_index" in z:
        ev["quartz_index"] = _pos(z["quartz_index"])
    for k in ("hyper_clay", "hyper_iron"):
        if k in raw and np.isfinite(raw[k]).any():
            # absolute band depth: 0.02 = trace, 0.12 or deeper = strong (membership 1)
            ev[k] = np.clip((np.nan_to_num(raw[k]) - 0.02) / 0.10, 0, 1)
            ev["hyper_cover"] = np.isfinite(raw[k]) | ev.get("hyper_cover", False)
    for k in ev:
        if k != "hyper_cover":
            ev[k] = np.where(usable, ev[k], 0).astype(np.float32)
    return ev


def model_scores(ev: dict[str, np.ndarray]) -> np.ndarray:
    """(n_models, H, W) prospectivity 0-100 (0.7 weighted membership = 100)."""
    out = []
    for m in MINERAL_MODELS:
        w = VEIN_WEIGHTS_WITH_ASTER if m.key == "vein" and "quartz_index" in ev else m.weights
        s = sum(wt * ev[k] for k, wt in w.items()) / sum(w.values())
        hw = HYPER_WEIGHTS.get(m.key)
        if hw and all(k in ev for k in hw) and "hyper_cover" in ev:
            s_h = sum(wt * ev[k] for k, wt in hw.items()) / sum(hw.values())
            s = np.where(ev["hyper_cover"], s_h, s)
        if m.key == "vein":   # structures alone are everywhere in the Karakoram: require the host rock too
            host = ev["quartz_index"] if "quartz_index" in ev else ev["silica"]
            s = s * np.clip(0.4 + host, 0, 1)
        out.append(np.clip(ndimage.uniform_filter(s, 3) * 100 / 0.7, 0, 100))
    return np.stack(out).astype(np.float32)


def find_targets(scores: np.ndarray, transform, pixel_area_ha: float, cutoffs: dict,
                 min_pixels: int = 12, elevation: Optional[np.ndarray] = None) -> list[dict]:
    targets = []
    for i, m in enumerate(MINERAL_MODELS):
        s = scores[i]
        mask = s >= cutoffs[m.key]
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if not n:
            continue
        ids = np.arange(1, n + 1)
        size = ndimage.sum(mask, lab, ids)
        mean = ndimage.mean(s, lab, ids)
        peak = ndimage.maximum(s, lab, ids)
        com = ndimage.center_of_mass(mask, lab, ids)
        elev = ndimage.mean(np.nan_to_num(elevation), lab, ids) if elevation is not None else [None] * n
        for sz, mn, pk, (r, c), el in zip(size, mean, peak, com, elev):
            if sz < min_pixels:
                continue
            x, y = transform * (c + 0.5, r + 0.5)
            targets.append({"model": m.key, "model_name": m.name, "commodities": m.commodities,
                            "x": float(x), "y": float(y), "pixels": int(sz),
                            "area_ha": round(float(sz * pixel_area_ha), 2), "mean_score": round(float(mn), 1),
                            "peak_score": round(float(pk), 1),
                            "elevation_m": round(float(el), 0) if el is not None else None,
                            "rank_score": round(float(mn) * float(np.log10(10 + sz)), 2)})
    return targets


def target_cutoffs(background: dict[str, np.ndarray]) -> dict[str, float]:
    cut = {}
    for m in MINERAL_MODELS:
        bg = background.get(m.key)
        p = float(np.percentile(bg, TARGET_PERCENTILE)) if bg is not None and bg.size else 100.0
        cut[m.key] = min(99.0, max(TARGET_THRESHOLD, p))
    return cut


# ----------------------------------------------------------------------------- known occurrences
def parse_occurrences(data: bytes, filename: str = "") -> list[dict]:
    """Known mineral occurrences (mines, prospects, showings) from CSV or GeoJSON points.

    CSV columns: lat, lon, commodity[, name]. Returns [{"lat","lon","commodity","model","name"}];
    commodities that no model covers are kept with model None (they are listed but not validated).
    """
    text = data.decode("utf-8-sig")
    out = []
    if filename.lower().endswith((".geojson", ".json")) or text.lstrip().startswith("{"):
        gj = json.loads(text)
        for f in gj.get("features", [gj] if gj.get("type") == "Feature" else []):
            g = f.get("geometry") or {}
            if g.get("type") != "Point":
                continue
            p = f.get("properties") or {}
            lon, lat = g["coordinates"][:2]
            out.append({"lat": float(lat), "lon": float(lon),
                        "commodity": str(p.get("commodity") or p.get("mineral") or "").strip().lower(),
                        "name": str(p.get("name") or p.get("locality") or "")})
    else:
        for row in csv.DictReader(io.StringIO(text)):
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            lat = row.get("lat") or row.get("latitude") or row.get("y")
            lon = row.get("lon") or row.get("lng") or row.get("longitude") or row.get("x")
            if not lat or not lon:
                continue
            out.append({"lat": float(lat), "lon": float(lon),
                        "commodity": (row.get("commodity") or row.get("mineral") or "").lower(),
                        "name": row.get("name") or row.get("locality") or ""})
    for o in out:
        if not (-90 <= o["lat"] <= 90 and -180 <= o["lon"] <= 180):
            raise ValueError(f"invalid coordinates: {o['lat']}, {o['lon']}")
        o["model"] = COMMODITY_TYPES.get(o["commodity"])
    return out
