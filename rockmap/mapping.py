"""Map generation: colour-coded lithology maps, image composites, statistics and figures."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage

from .config import ALL_CLASSES, CLASS_BY_ID, CLASS_IDS, CLOUD_CLASS, MASK_CLASSES, class_name, hex_to_rgb


def majority_filter(labels: np.ndarray, size: int = 3, keep: Optional[np.ndarray] = None) -> np.ndarray:
    """Modal filter to remove isolated 'salt and pepper' pixels from a classified map."""
    if size <= 1:
        return labels
    classes = [c for c in CLASS_IDS if (labels == c).any()]
    if not classes:
        return labels
    votes = np.stack([ndimage.uniform_filter((labels == c).astype(np.float32), size) for c in classes])
    out = np.asarray(classes, dtype=labels.dtype)[votes.argmax(0)]
    masked = (labels == 0) | (labels >= 250)
    out[masked] = labels[masked]
    if keep is not None:
        out[keep] = labels[keep]
    return out


def colorize(labels: np.ndarray) -> np.ndarray:
    """(H, W) class ids -> (H, W, 4) RGBA uint8. Cloud / no-data pixels are transparent."""
    lut = np.zeros((256, 4), np.uint8)
    for cid, c in ALL_CLASSES.items():
        lut[cid] = (*hex_to_rgb(c.color), 255)
    lut[CLOUD_CLASS] = 0
    return lut[labels.astype(np.uint8)]


def stretch(band: np.ndarray, valid: Optional[np.ndarray] = None, lo: float = 2, hi: float = 98) -> np.ndarray:
    vals = band[valid] if valid is not None and valid.any() else band[np.isfinite(band)]
    if vals.size == 0:
        return np.zeros(band.shape, np.uint8)
    a, b = np.percentile(vals, [lo, hi])
    return (np.clip((np.nan_to_num(band) - a) / (b - a + 1e-9), 0, 1) * 255).astype(np.uint8)


def composite(refl: np.ndarray, bands=(2, 1, 0), valid: Optional[np.ndarray] = None) -> np.ndarray:
    """RGB composite with a 2-98 % stretch. Default true colour (red, green, blue).

    ``bands=(5, 3, 0)`` (SWIR2, NIR, blue) is a common geology false-colour composite.
    """
    return np.dstack([stretch(refl[b], valid) for b in bands])


def save_png(arr: np.ndarray, path: str | Path, max_size: Optional[int] = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr)
    if max_size and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.NEAREST if arr.shape[-1] == 4 else Image.BILINEAR)
    img.save(path, optimize=True)
    return path


def hillshade_png(dem: np.ndarray, pixel_size: float = 20.0) -> np.ndarray:
    dy, dx = np.gradient(dem.astype(np.float64), pixel_size)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az, alt = np.radians(315), np.radians(45)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    g = (np.clip(hs, 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def area_statistics_from_counts(counts: np.ndarray, pixel_area_km2: float) -> list[dict]:
    """Area of each lithology (percent of classified rock) and of each masked surface type."""
    counts = np.asarray(counts)
    rock_total = int(sum(counts[c] for c in CLASS_IDS))
    stats = []
    for cid in CLASS_IDS:
        n = int(counts[cid])
        stats.append({"id": cid, "name": class_name(cid), "color": CLASS_BY_ID[cid].color, "pixels": n,
                      "area_km2": n * pixel_area_km2, "percent": 100.0 * n / rock_total if rock_total else 0.0,
                      "kind": "rock"})
    for m in MASK_CLASSES:
        n = int(counts[m.id])
        if n:
            stats.append({"id": m.id, "name": m.name, "color": m.color, "pixels": n,
                          "area_km2": n * pixel_area_km2, "percent": None, "kind": "mask"})
    return stats


def area_statistics(labels: np.ndarray, pixel_area_km2: float) -> list[dict]:
    """Area and percentage of each lithology class in a classified map."""
    return area_statistics_from_counts(np.bincount(labels.astype(np.uint8).ravel(), minlength=256),
                                       pixel_area_km2)


def plot_confusion(metrics: dict, path: str | Path, title: str = "Confusion matrix") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    names = [class_name(c).split(" /")[0].split(" (")[0] for c in metrics["classes"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=110)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    ax.set_title(f"{title}\nOA {metrics['overall_accuracy'] * 100:.1f}%  kappa {metrics['kappa']:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_map_with_legend(labels: np.ndarray, path: str | Path, title: str = "Lithological map") -> Path:
    """Publication-style map figure with legend (for the project report)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=110)
    ax.imshow(colorize(labels), interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    present = [c for c in list(CLASS_IDS) + [m.id for m in MASK_CLASSES if m.id != CLOUD_CLASS]
               if (labels == c).any()]
    handles = [Patch(facecolor=ALL_CLASSES[c].color, edgecolor="#333", label=ALL_CLASSES[c].name) for c in present]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
