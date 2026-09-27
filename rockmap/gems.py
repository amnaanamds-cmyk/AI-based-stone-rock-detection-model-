"""Gemstone prospectivity mapping.

Gem crystals are millimetres to centimetres across and cannot be seen in 10-20 m satellite
pixels. What *can* be mapped from space are the host rocks and geological settings in which
gems form. RockMap combines Sentinel-2 spectral evidence and DEM-derived texture into four
knowledge-driven deposit models that cover the main gem settings of Gilgit-Baltistan and the
wider Himalaya-Karakoram:

======================  =============================================  ==========================
model                   gems                                           host setting (GB examples)
======================  =============================================  ==========================
``marble``              ruby, spinel, pargasite                        marble bands in metamorphic
                                                                       belts (central Hunza)
``pegmatite``           aquamarine, topaz, tourmaline, garnet,         granitic / leucogranite
                        quartz crystals                                pegmatites (Shigar, Braldu,
                                                                       Haramosh, Nagar)
``contact``             emerald, beryl                                 pegmatites cutting mafic /
                                                                       ultramafic schist (Khaltaro)
``ultramafic``          peridot, nephrite jade, serpentine             ultramafic & ophiolitic rocks
                                                                       (Kohistan arc, sutures)
======================  =============================================  ==========================

With known gem occurrences (uploaded localities or field-app finds) the models are validated
(success rate / AUC: are known occurrences in the top-ranked ground?) and, when enough
occurrences exist, a data-driven Random Forest prospectivity model is trained on them.

All results are *screening* layers that tell field teams where to look first.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

EPS = 1e-6

GEM_TYPES = {
    "ruby": "marble", "spinel": "marble", "pargasite": "marble",
    "aquamarine": "pegmatite", "topaz": "pegmatite", "tourmaline": "pegmatite", "garnet": "pegmatite",
    "quartz": "pegmatite", "beryl": "contact", "emerald": "contact", "peridot": "ultramafic",
    "nephrite": "ultramafic", "serpentine": "ultramafic",
}
GEM_NAMES = sorted(GEM_TYPES)


@dataclass(frozen=True)
class GemModel:
    key: str
    name: str
    gems: str
    color: str
    setting: str
    weights: dict


GEM_MODELS: tuple[GemModel, ...] = (
    GemModel("marble", "Marble-hosted ruby & spinel", "ruby, spinel, pargasite", "#d6254d",
             "Bright carbonate (CO3 absorption, high SWIR1/SWIR2, low iron) in banded metamorphic terrain, "
             "near dark mafic / schist contacts.",
             {"carbonate": 0.55, "carb_contact": 0.25, "texture": 0.20}),
    GemModel("pegmatite", "Pegmatite gems", "aquamarine, topaz, tourmaline, garnet, quartz", "#2b8cbe",
             "Bright, iron-poor felsic bodies (leucogranite / pegmatite swarms) with dyke-like texture, "
             "intruding darker metamorphic host rocks.",
             {"felsic": 0.50, "texture": 0.25, "felsic_contact": 0.25}),
    GemModel("contact", "Emerald & beryl contacts", "emerald, beryl", "#1a9850",
             "Felsic pegmatite in contact with mafic / ultramafic schist (Be meets Cr - Khaltaro type).",
             {"felsic_contact": 0.60, "mafic": 0.20, "felsic": 0.20}),
    GemModel("ultramafic", "Ultramafic-hosted gems", "peridot, nephrite jade, serpentine", "#6a51a3",
             "Dark, ferrous (olivine / pyroxene) rocks with Mg-OH absorption (serpentinisation).",
             {"mafic": 0.60, "mgoh": 0.40}),
)
MODEL_BY_KEY = {m.key: m for m in GEM_MODELS}
TARGET_THRESHOLD = 75       # absolute score a candidate zone must reach
TARGET_PERCENTILE = 99.0    # ...and it must rank in the top 1 % of the region for that deposit model
MAX_TARGETS_PER_MODEL = 50
EVIDENCE = ("brightness", "iron", "ferrous", "swir_ratio", "texture")


def raw_evidence(refl: np.ndarray, cos_i: Optional[np.ndarray] = None,
                 usable: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Per-pixel spectral quantities from a 6-band reflectance stack (blue..swir2).

    Band *ratios* are nearly independent of illumination, but brightness is not: sun-facing
    slopes look brighter. When ``cos_i`` (solar incidence from the DEM) is given, the part of the
    brightness explained by illumination is removed (per-scene linear fit), so bright *rock* is
    found instead of bright *slopes*. Texture is computed on that illumination-free brightness.
    """
    r = np.nan_to_num(refl[:6])
    blue, green, red, nir, swir1, swir2 = r
    bright = r.mean(axis=0)
    if cos_i is not None:
        ok = np.isfinite(cos_i) & (bright > 0) & (usable if usable is not None else True)
        if ok.sum() > 500:
            b, a = np.polyfit(cos_i[ok].astype(np.float64), bright[ok].astype(np.float64), 1)
            bright = bright - (a + b * np.nan_to_num(cos_i)) + float(np.median(bright[ok]))
    return {
        "brightness": bright,
        "iron": red / (blue + EPS),                                 # ferric iron staining
        "ferrous": swir2 / (nir + EPS) + green / (red + EPS),       # Fe2+ (mafic minerals)
        "swir_ratio": swir1 / (swir2 + EPS),                        # CO3 / Mg-OH / Al-OH absorption at 2.2-2.35 um
        "texture": ndimage.uniform_filter(bright ** 2, 5) - ndimage.uniform_filter(bright, 5) ** 2,
    }


@dataclass
class EvidenceStats:
    median: dict
    scale: dict

    def z(self, key: str, v: np.ndarray) -> np.ndarray:
        return (v - self.median[key]) / self.scale[key]

    def to_dict(self) -> dict:
        return {"median": self.median, "scale": self.scale}

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceStats":
        return cls(d["median"], d["scale"])

    @classmethod
    def fit(cls, samples: dict[str, np.ndarray]) -> "EvidenceStats":
        med, sc = {}, {}
        for k, v in samples.items():
            v = v[np.isfinite(v)]
            m = float(np.median(v)) if v.size else 0.0
            mad = float(np.median(np.abs(v - m))) * 1.4826 if v.size else 1.0
            med[k], sc[k] = m, max(mad, 1e-4)
        return cls(med, sc)


def _pos(z: np.ndarray) -> np.ndarray:
    """Positive anomaly mapped to 0..1 (3 robust sigma = 1)."""
    return np.clip(z / 3.0, 0, 1)


def _contact(a: np.ndarray, b: np.ndarray, pixel_size: float, scale_m: float = 120.0) -> np.ndarray:
    """High where pixels of type a lie within a few hundred metres of type b."""
    if not a.any() or not b.any():
        return np.zeros(a.shape, np.float32)
    da = ndimage.distance_transform_edt(~a) * pixel_size
    db = ndimage.distance_transform_edt(~b) * pixel_size
    return np.exp(-(da + db) / (2 * scale_m)).astype(np.float32)


def bedrock_mask(dem: Optional[np.ndarray], pixel_size: float, min_slope: float = 15.0) -> Optional[np.ndarray]:
    """Exposed bedrock proxy: slopes >= ``min_slope`` degrees. Gem host rocks crop out on valley walls,
    not on alluvial fans, terraces or river beds (which are bright and would otherwise look felsic)."""
    if dem is None or not np.isfinite(dem).any():
        return None
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem)).astype(np.float64)
    gy, gx = np.gradient(d, pixel_size)
    return np.degrees(np.arctan(np.hypot(gx, gy))) >= min_slope


def evidence_layers(refl: np.ndarray, usable: np.ndarray, stats: EvidenceStats, pixel_size: float = 20.0,
                    lithology: Optional[np.ndarray] = None, dem: Optional[np.ndarray] = None,
                    cos_i: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
    """Normalised (0..1) evidence maps used by the deposit models."""
    raw = raw_evidence(refl, cos_i, usable)
    z = {k: stats.z(k, raw[k]) for k in EVIDENCE}
    swir, bright, iron, ferr = _pos(z["swir_ratio"]), _pos(z["brightness"]), _pos(z["iron"]), _pos(z["ferrous"])
    not_dark = np.clip((z["brightness"] + 1) / 2, 0, 1)          # at or above median brightness
    ev = {
        # marble / carbonate: a real 2.33 um CO3 absorption (SWIR1/SWIR2) on bright, iron-poor rock
        # (a weak feature at Sentinel-2 resolution: B12 only covers the shoulder of the 2.33 um band,
        # so 2 bedrock sigma already counts as a full anomaly)
        "carbonate": np.clip(z["swir_ratio"] / 2.0, 0, 1) * not_dark * (1 - iron),
        # leucogranite / pegmatite: bright, spectrally FLAT (no SWIR absorption), iron- and Fe2+-poor
        "felsic": bright * (1 - swir) * (1 - ferr) * (1 - iron),
        # mafic / ultramafic: dark and ferrous
        "mafic": ferr * np.clip((-z["brightness"] + 1) / 2, 0, 1),
        # Mg-OH (serpentine, chlorite, talc): SWIR2 absorption on dark rock
        "mgoh": swir * np.clip((-z["brightness"] + 1) / 2, 0, 1),
        "texture": _pos(z["texture"]),
    }
    rock = bedrock_mask(dem, pixel_size)
    ok = usable if rock is None else usable & rock
    if lithology is not None:   # a lithology map, when available, sharpens the spectral evidence
        ev["carbonate"] = np.maximum(ev["carbonate"], 0.6 * (lithology == 1))
        ev["felsic"] = np.maximum(ev["felsic"], 0.4 * (lithology == 4))
        ev["mafic"] = np.maximum(ev["mafic"], 0.5 * (lithology == 5))
        ok = ok & (lithology != 7)          # no gems in Quaternary sediments (alluvium, moraine, scree)
    strong = {k: (ev[k] > 0.6) & ok for k in ("carbonate", "felsic", "mafic")}
    ev["carb_contact"] = _contact(strong["carbonate"], strong["mafic"], pixel_size, 80.0)
    ev["felsic_contact"] = _contact(strong["felsic"], strong["mafic"], pixel_size, 80.0)
    for k in ev:
        ev[k] = np.where(ok, ev[k], 0).astype(np.float32)
    return ev


def model_scores(ev: dict[str, np.ndarray]) -> np.ndarray:
    """(n_models, H, W) prospectivity scores 0-100."""
    out = []
    for m in GEM_MODELS:
        s = sum(w * ev[k] for k, w in m.weights.items()) / sum(m.weights.values())
        out.append(np.clip(ndimage.uniform_filter(s, 3) * 100 / 0.7, 0, 100))   # 0.7 weighted evidence = 100
    return np.stack(out).astype(np.float32)


def find_gem_targets(scores: np.ndarray, transform, pixel_area_ha: float, threshold: float = 60.0,
                     min_pixels: int = 12, elevation: Optional[np.ndarray] = None) -> list[dict]:
    """Connected candidate areas for every model (small: pegmatite swarms are narrow). A low
    ``threshold`` is used here; :func:`rank_targets` keeps only regionally outstanding ones."""
    targets = []
    for i, m in enumerate(GEM_MODELS):
        s = scores[i]
        mask = s >= threshold
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
            x, y = transform @ (c + 0.5, r + 0.5)
            targets.append({"model": m.key, "model_name": m.name, "gems": m.gems, "x": float(x), "y": float(y),
                            "pixels": int(sz), "area_ha": round(float(sz * pixel_area_ha), 2),
                            "mean_score": round(float(mn), 1), "peak_score": round(float(pk), 1),
                            "elevation_m": round(float(el), 0) if el is not None else None,
                            "rank_score": round(float(mn) * float(np.log10(10 + sz)), 2)})
    return targets


def target_cutoffs(background: dict[str, np.ndarray]) -> dict[str, float]:
    """Per-model score above which ground is a *target*: at least TARGET_THRESHOLD and within the
    top (100 - TARGET_PERCENTILE) % of the region. Relative ranking is how prospectivity maps are
    used - field teams visit the most prospective ground first - and it keeps target zones compact."""
    cut = {}
    for m in GEM_MODELS:
        bg = background.get(m.key)
        p = float(np.percentile(bg, TARGET_PERCENTILE)) if bg is not None and bg.size else 100.0
        cut[m.key] = min(99.0, max(TARGET_THRESHOLD, p))
    return cut


# ----------------------------------------------------------------------------- known occurrences
def parse_occurrences(data: bytes, filename: str = "") -> list[dict]:
    """Known gem localities from GeoJSON points or CSV (lat, lon, gem[, name]).

    Returns [{"lat", "lon", "gem", "model", "name"}]; unknown gem names are kept with model None.
    """
    text = data.decode("utf-8-sig")
    out = []
    if filename.lower().endswith((".geojson", ".json")) or text.lstrip().startswith("{"):
        gj = json.loads(text)
        feats = gj.get("features", [gj] if gj.get("type") == "Feature" else [])
        for f in feats:
            g = f.get("geometry") or {}
            if g.get("type") != "Point":
                continue
            p = f.get("properties") or {}
            lon, lat = g["coordinates"][:2]
            out.append({"lat": float(lat), "lon": float(lon), "gem": str(p.get("gem") or p.get("mineral") or "").lower(),
                        "name": str(p.get("name") or p.get("locality") or "")})
    else:
        rows = csv.DictReader(io.StringIO(text))
        for row in rows:
            row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
            lat = row.get("lat") or row.get("latitude") or row.get("y")
            lon = row.get("lon") or row.get("lng") or row.get("longitude") or row.get("x")
            if not lat or not lon:
                continue
            out.append({"lat": float(lat), "lon": float(lon), "gem": (row.get("gem") or row.get("mineral") or "").lower(),
                        "name": row.get("name") or row.get("locality") or ""})
    for o in out:
        if not (-90 <= o["lat"] <= 90 and -180 <= o["lon"] <= 180):
            raise ValueError(f"invalid coordinates: {o['lat']}, {o['lon']}")
        o["model"] = GEM_TYPES.get(o["gem"].rstrip("s"))
    return out


def success_rates(occ_scores: list[float], background: np.ndarray) -> dict:
    """Validation of a prospectivity map with known occurrences.

    For each occurrence the percentile of its score in the background score distribution is
    computed. AUC = mean percentile (0.5 = random, 1 = perfect). ``top10`` / ``top20`` = share of
    occurrences that fall in the 10 % / 20 % most prospective ground.
    """
    bg = np.sort(background[np.isfinite(background)])
    if not len(occ_scores) or not bg.size:
        return {"n": 0}
    pct = np.searchsorted(bg, np.asarray(occ_scores), side="right") / bg.size
    return {"n": int(len(occ_scores)), "auc": round(float(pct.mean()), 3),
            "top10": round(float((pct >= 0.9).mean()), 3), "top20": round(float((pct >= 0.8).mean()), 3)}


FEATURE_KEYS = ("carbonate", "felsic", "mafic", "mgoh", "texture", "carb_contact", "felsic_contact")
