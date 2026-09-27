"""Model-free geological analytics that work directly on the imagery and DEM.

* **Mineral alteration / prospectivity** - hydrothermal alteration leaves clay minerals
  (Al-OH / Mg-OH absorption at 2.2 um, high SWIR1/SWIR2) and iron oxides (gossans, high
  red/blue) at the surface. Robust region-wide anomalies of these Sentinel-2 ratios highlight
  exploration targets (classic band-ratio method, Sabins 1999; van der Meer et al. 2014).
* **Spectral units** - unsupervised k-means clustering of spectra + indices separates
  surface materials without any training data; a geologist then names the clusters.
* **Landslide / rockfall susceptibility** - knowledge-driven weighted overlay of slope,
  local relief, proximity to rivers, rock strength and bare ground.

All outputs are screening tools and are labelled as such in the dashboard and reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage

EPS = 1e-6

# ----------------------------------------------------------------------------- alteration
ALTERATION_INDICES = ("clay", "iron_oxide", "ferrous", "gossan")
ALTERATION_LABELS = {
    "clay": "Clay / hydroxyl (argillic, phyllic)",
    "iron_oxide": "Iron oxide (ferric, gossan)",
    "ferrous": "Ferrous iron (mafic, propylitic)",
    "gossan": "Gossan (SWIR1 / red)",
}
ALTERATION_WEIGHTS = {"clay": 0.4, "iron_oxide": 0.35, "ferrous": 0.15, "gossan": 0.10}
TARGET_THRESHOLD = 75       # score (0-100) above which a pixel is anomalous (~2.25 robust sigma)


def alteration_indices(refl: np.ndarray) -> dict[str, np.ndarray]:
    """Sentinel-2 alteration ratios from a canonical 6-band stack (blue..swir2)."""
    blue, green, red, nir, swir1, swir2 = (np.nan_to_num(b) for b in refl[:6])
    return {
        "clay": swir1 / (swir2 + EPS),
        "iron_oxide": red / (blue + EPS),
        "ferrous": swir2 / (nir + EPS) + green / (red + EPS),
        "gossan": swir1 / (red + EPS),
    }


@dataclass
class RobustStats:
    median: dict
    scale: dict

    def to_dict(self) -> dict:
        return {"median": self.median, "scale": self.scale}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustStats":
        return cls(d["median"], d["scale"])

    @classmethod
    def fit(cls, samples: dict[str, np.ndarray]) -> "RobustStats":
        med, sc = {}, {}
        for k, v in samples.items():
            v = v[np.isfinite(v)]
            m = float(np.median(v)) if v.size else 0.0
            mad = float(np.median(np.abs(v - m))) * 1.4826 if v.size else 1.0
            med[k], sc[k] = m, max(mad, 1e-3)
        return cls(med, sc)


def alteration_score(refl: np.ndarray, usable: np.ndarray, stats: RobustStats) -> tuple[np.ndarray, np.ndarray]:
    """Returns (score 0-100 float32, dominant index id uint8 1..4; 0 where not usable).

    Ratios are unstable on dark pixels and on partly snow-covered pixels, so those are excluded:
    mean reflectance < 0.08 or NDSI > 0.15 (sub-pixel snow / ice). Sparse vegetation and alpine
    meadows raise the SWIR-based ratios, so clay / ferrous / gossan anomalies are ignored where
    NDVI > 0.25 (the red/blue iron-oxide ratio is hardly affected by vegetation).
    """
    r = np.nan_to_num(refl[:6])
    ndsi = (r[1] - r[4]) / (r[1] + r[4] + EPS)
    ndvi = (r[3] - r[2]) / (r[3] + r[2] + EPS)
    usable = usable & (r.mean(axis=0) >= 0.08) & (ndsi <= 0.15)
    idx = alteration_indices(refl)
    z = np.stack([np.clip((idx[k] - stats.median[k]) / stats.scale[k], 0, 10) for k in ALTERATION_INDICES])
    veg = ndvi > 0.25
    for i, k in enumerate(ALTERATION_INDICES):
        if k != "iron_oxide":
            z[i][veg] = 0
    w = np.asarray([ALTERATION_WEIGHTS[k] for k in ALTERATION_INDICES], np.float32)[:, None, None]
    combined = (z * w).sum(0) / w.sum()
    # 3 robust sigma of weighted anomaly -> 100; smooth slightly to suppress single-pixel noise
    score = np.clip(ndimage.uniform_filter(combined, 3) / 3.0 * 100, 0, 100).astype(np.float32)
    dominant = (np.argmax(z * w, axis=0) + 1).astype(np.uint8)
    score[~usable] = 0
    dominant[~usable] = 0
    return score, dominant


def find_targets(score: np.ndarray, dominant: np.ndarray, transform, threshold: float = TARGET_THRESHOLD,
                 min_pixels: int = 25, pixel_area_ha: float = 0.04, extra: Optional[dict] = None) -> list[dict]:
    """Connected anomalous areas -> candidate targets with centroid (map coords), size and score."""
    mask = score >= threshold
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return []
    ids = np.arange(1, n + 1)
    size = ndimage.sum(mask, lab, ids)
    mean = ndimage.mean(score, lab, ids)
    peak = ndimage.maximum(score, lab, ids)
    com = ndimage.center_of_mass(mask, lab, ids)
    out = []
    for i, (sz, mn, pk, (r, c)) in enumerate(zip(size, mean, peak, com)):
        if sz < min_pixels:
            continue
        comp = lab == ids[i]
        dom = np.bincount(dominant[comp], minlength=5)[1:]
        kind = ALTERATION_INDICES[int(np.argmax(dom))]
        x, y = transform @ (c + 0.5, r + 0.5)
        t = {"x": float(x), "y": float(y), "pixels": int(sz), "area_ha": round(float(sz * pixel_area_ha), 2),
             "mean_score": round(float(mn), 1), "peak_score": round(float(pk), 1), "type": kind,
             "type_label": ALTERATION_LABELS[kind]}
        for k, arr in (extra or {}).items():
            vals = arr[comp]
            vals = vals[np.isfinite(vals)] if np.issubdtype(vals.dtype, np.floating) else vals
            if vals.size:
                t[k] = round(float(np.mean(vals)), 1) if np.issubdtype(vals.dtype, np.floating) \
                    else int(np.bincount(vals.astype(np.int64)).argmax())
        t["rank_score"] = round(float(mn) * np.log10(10 + sz), 2)
        out.append(t)
    return out


# ----------------------------------------------------------------------------- landslide susceptibility
HAZARD_CLASSES = {1: ("Very low", "#1a9850"), 2: ("Low", "#91cf60"), 3: ("Moderate", "#fee08b"),
                  4: ("High", "#fc8d59"), 5: ("Very high", "#d73027")}
# relative weakness (0 strong .. 1 weak) of each lithology class for slope failure
LITHO_WEAKNESS = {1: 0.4, 2: 0.5, 3: 0.8, 4: 0.3, 5: 0.35, 6: 0.6, 7: 0.9}
HAZARD_WEIGHTS = {"slope": 0.40, "relief": 0.15, "river": 0.15, "lithology": 0.20, "bare": 0.10}


def landslide_susceptibility(dem: np.ndarray, pixel_size: float, water: np.ndarray, ndvi: np.ndarray,
                             lithology: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-overlay susceptibility index (0-1) and 5 classes (1 very low .. 5 very high)."""
    d = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0).astype(np.float64)
    gy, gx = np.gradient(d, pixel_size)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    f_slope = np.interp(slope, [0, 10, 20, 30, 40, 50, 90], [0.0, 0.1, 0.4, 0.7, 0.95, 1.0, 0.9])
    size = max(3, int(round(300 / pixel_size)) | 1)
    relief = ndimage.maximum_filter(d, size) - ndimage.minimum_filter(d, size)
    f_relief = np.clip(relief / 400.0, 0, 1)
    if water.any():
        dist = ndimage.distance_transform_edt(~water) * pixel_size
        f_river = np.exp(-dist / 400.0)            # undercutting by rivers (e.g. Hunza, Indus, Gilgit)
    else:
        f_river = np.zeros_like(d)
    if lithology is not None and np.isin(lithology, list(LITHO_WEAKNESS)).any():
        lut = np.full(256, 0.5)
        for k, v in LITHO_WEAKNESS.items():
            lut[k] = v
        f_lith = lut[lithology.astype(np.uint8)]
    else:
        f_lith = np.full_like(d, 0.5)
    f_bare = 1 - np.clip((np.nan_to_num(ndvi) - 0.1) / 0.5, 0, 0.7)
    w = HAZARD_WEIGHTS
    index = (w["slope"] * f_slope + w["relief"] * f_relief + w["river"] * f_river + w["lithology"] * f_lith
             + w["bare"] * f_bare) / sum(w.values())
    classes = np.digitize(index, [0.35, 0.50, 0.65, 0.80]).astype(np.uint8) + 1
    return index.astype(np.float32), classes


# ----------------------------------------------------------------------------- spectral units
CLUSTER_PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                   "#bcbd22", "#17becf", "#393b79", "#ad494a", "#637939", "#e7ba52", "#7b4173", "#3182bd"]


def spectral_features(refl: np.ndarray) -> np.ndarray:
    """Bands + indices used for clustering (brightness-normalised so shading matters less)."""
    from .features import spectral_indices
    r = np.nan_to_num(refl[:6])
    bright = r.mean(axis=0, keepdims=True) + EPS
    shape = r / bright                                  # spectral shape independent of illumination
    return np.concatenate([shape, np.log(bright), spectral_indices(r)], axis=0).astype(np.float32)


def fit_clusters(samples: np.ndarray, k: int = 10, seed: int = 0):
    """samples: (n, f) spectral features -> fitted (scaler mean, std, KMeans)."""
    from sklearn.cluster import MiniBatchKMeans
    mean = samples.mean(0)
    std = samples.std(0) + 1e-6
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=5, batch_size=4096, max_iter=200)
    km.fit((samples - mean) / std)
    return mean.astype(np.float32), std.astype(np.float32), km


def predict_clusters(feats: np.ndarray, usable: np.ndarray, mean, std, km) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel cluster (1..k, 0 = not usable) and a 0-1 membership confidence (margin between the
    nearest and second-nearest cluster centre)."""
    f, h, w = feats.shape
    labels = np.zeros((h, w), np.uint8)
    conf = np.zeros((h, w), np.float32)
    idx = np.nonzero(usable)
    if len(idx[0]):
        x = (feats[:, idx[0], idx[1]].T - mean) / std
        d = km.transform(x)
        order = np.sort(d, axis=1)
        labels[idx] = (np.argmin(d, axis=1) + 1).astype(np.uint8)
        conf[idx] = np.clip(1 - order[:, 0] / (order[:, 1] + EPS), 0, 1) * 2
    return labels, np.clip(conf, 0, 1)


# ----------------------------------------------------------------------------- insights
def insights(stats: Optional[dict], analytics: Optional[dict], gems: Optional[dict] = None) -> list[str]:
    """Plain-language findings for the dashboard and the executive summary of the report."""
    out: list[str] = []
    if stats and stats.get("classified_km2"):
        rock = sorted([s for s in stats["region"] if s["id"] < 250 and s["pixels"]], key=lambda s: -s["area_km2"])
        if rock:
            top = rock[0]
            out.append(f"{top['name']} is the dominant rock type, covering {top['area_km2']:,.0f} km² "
                       f"({top['percent']:.0f}% of mapped rock).")
            if len(rock) > 2:
                out.append("The next most extensive units are " + " and ".join(
                    f"{r['name']} ({r['percent']:.0f}%)" for r in rock[1:3]) + ".")
        snow = next((s for s in stats["region"] if s["id"] == 250), None)
        if snow and snow["pixels"]:
            out.append(f"Snow and glaciers cover {snow['area_km2']:,.0f} km² of the mapped area and hide the rock beneath.")
        if stats.get("districts"):
            d = max(stats["districts"], key=lambda d: d["km2"])
            out.append(f"{len(stats['districts'])} districts analysed; the largest mapped district is {d['name']} "
                       f"({d['km2']:,.0f} km²).")
    if analytics:
        t = analytics.get("targets_total", 0)
        if t:
            top = analytics.get("top_target") or {}
            out.append(f"{t} mineral-alteration anomalies were detected; the strongest "
                       f"({top.get('type_label', '')}, score {top.get('mean_score', 0):.0f}) lies at "
                       f"{top.get('lat', 0):.4f}°N, {top.get('lon', 0):.4f}°E.")
        hz = analytics.get("hazard_km2") or {}
        high = hz.get("4", 0) + hz.get("5", 0)
        total = sum(hz.values())
        if total:
            out.append(f"{high:,.0f} km² ({100 * high / total:.0f}%) of the assessed terrain has high or very high "
                       "landslide / rockfall susceptibility.")
        if analytics.get("clusters"):
            out.append(f"{len(analytics['clusters'])} spectral units were identified without training data; "
                       f"{sum(1 for c in analytics['clusters'] if c.get('class_id'))} have been named by a geologist.")
    if gems and gems.get("targets_total"):
        best = max(gems["models"], key=lambda m: m["targets"])
        top = gems["targets_top"][0]
        out.append(f"{gems['targets_total']} gemstone target zones were ranked; the most numerous setting is "
                   f"{best['name'].lower()} ({best['targets']} zones). Top zone: {top['model_name']} "
                   f"({top['gems']}) at {top['lat']:.4f}°N, {top['lon']:.4f}°E.")
        v = (gems.get("validation") or {}).get("overall") or {}
        if v.get("n"):
            out.append(f"{v['n']} known gem localities check the map: {v['top20'] * 100:.0f}% lie in the 20% most "
                       f"prospective ground (AUC {v['auc']:.2f}; 0.5 = random).")
    return out
