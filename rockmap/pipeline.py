"""End-to-end pipeline: load data -> features -> train/benchmark -> classify -> validate -> map."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window

from . import __version__
from .config import (ALGORITHM_LABELS, ALGORITHMS, CLASS_IDS, CLOUD_CLASS, DEFAULT_BLOCK_SIZE,
                     DEFAULT_PATCH_SIZE, DEFAULT_SAMPLES_PER_CLASS)
from .evaluation import compare_maps, evaluate
from .features import Normalizer, build_features
from .io import GeoInfo, align_to, class_colormap, raster_shape, read_raster, write_raster
from .mapping import (area_statistics, colorize, composite, hillshade_png, majority_filter,
                      plot_confusion, plot_map_with_legend, save_png)
from .models import ClassicalClassifier, CNNClassifier
from .preprocessing import preprocess
from .sampling import block_split, extract_patches, pixel_vectors, sample_pixels

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


@dataclass
class SceneData:
    reflectance: np.ndarray   # (6, H, W)
    valid: np.ndarray         # (H, W) bool
    dem: Optional[np.ndarray]  # (H, W) metres
    info: GeoInfo

    @property
    def pixel_size_m(self) -> float:
        px = self.info.pixel_size[0]
        if self.info.crs is not None and self.info.crs.is_geographic:
            return px * 111_320.0
        return px or 1.0


def parse_window(window: Optional[Sequence[int]]) -> Optional[Window]:
    """(col_off, row_off, width, height) -> rasterio Window."""
    if window is None:
        return None
    x, y, w, h = (int(v) for v in window)
    return Window(x, y, w, h)


def load_scene(scene_path, dem_path=None, cloud_path=None, sensor: str = "reflectance",
               dos: bool = False, window: Optional[Sequence[int]] = None,
               cloud_band: Optional[int] = None) -> SceneData:
    """Read and preprocess a scene (and optional DEM / cloud layer) onto one grid.

    ``cloud_band`` selects a band inside the scene file itself holding a SCL / QA layer
    (as produced by ``rockmap stack --scl``).
    """
    win = parse_window(window)
    if win is not None:
        _, h, w = raster_shape(scene_path)
        win = win.intersection(Window(0, 0, w, h))
    raw, info = read_raster(scene_path, window=win)
    cloud_layer = None
    if cloud_band is not None:
        cloud_layer = raw[cloud_band - 1]
    elif raw.shape[0] > 6 and sensor != "reflectance":
        cloud_layer = raw[6]  # stacked SCL / QA band
    if cloud_path:
        cloud_layer = align_to(cloud_path, info, Resampling.nearest)
    refl, valid = preprocess(raw, sensor=sensor, dos=dos, cloud_layer=cloud_layer)
    dem = align_to(dem_path, info, Resampling.bilinear) if dem_path else None
    if dem is not None:
        valid &= np.isfinite(dem)
    return SceneData(refl, valid, dem, info)


def load_labels(label_path, info: GeoInfo) -> np.ndarray:
    """Reference geological map (class ids) resampled onto the scene grid."""
    lab = align_to(label_path, info, Resampling.nearest)
    return np.nan_to_num(lab, nan=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_models(scene_path, label_path, out_dir, dem_path=None, cloud_path=None,
                 algorithms: Sequence[str] = ALGORITHMS, sensor: str = "reflectance", dos: bool = False,
                 samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS, epochs: int = 30,
                 block_size: int = DEFAULT_BLOCK_SIZE, seed: int = 0, name: Optional[str] = None,
                 progress: Progress = _noop) -> dict:
    """Train the CNN and benchmark classifiers; write a model bundle to ``out_dir``.

    Bundle layout::

        meta.json        feature names, normaliser, classes, metrics for every algorithm
        cnn.pt           PyTorch CNN weights
        rf.joblib        Random Forest
        svm.joblib       Support Vector Machine
        cm_<algo>.png    confusion matrices on the held-out spatial test blocks
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    algorithms = [a for a in algorithms if a in ALGORITHMS]
    if not algorithms:
        raise ValueError(f"No valid algorithm given (choose from {', '.join(ALGORITHMS)})")
    progress(0.02, "Loading and preprocessing imagery")
    scene = load_scene(scene_path, dem_path, cloud_path, sensor, dos)
    labels = load_labels(label_path, scene.info)
    features, names = build_features(scene.reflectance, scene.dem, scene.pixel_size_m)

    progress(0.08, "Preparing training samples (spatial block split)")
    split = block_split(labels.shape, block_size, seed=seed)
    margin = DEFAULT_PATCH_SIZE // 2
    train = sample_pixels(labels, scene.valid, split, 0, samples_per_class, margin, seed)
    val = sample_pixels(labels, scene.valid, split, 1, max(500, samples_per_class // 4), margin, seed)
    test = sample_pixels(labels, scene.valid, split, 2, max(1000, samples_per_class // 2), margin, seed,
                         erode_boundaries=False)
    if len(train) < 50:
        raise ValueError("Too few labelled training pixels - check that the reference map overlaps the scene")
    classes = sorted(int(c) for c in np.unique(train.labels))

    norm = Normalizer().fit(pixel_vectors(features, train))
    feats_n = norm.transform_image(features)
    x_train, x_val, x_test = (pixel_vectors(feats_n, s) for s in (train, val, test))

    meta = {
        "name": name or out_dir.name,
        "version": __version__,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scene": str(scene_path),
        "labels": str(label_path),
        "sensor": sensor,
        "dos": dos,
        "uses_dem": scene.dem is not None,
        "feature_names": names,
        "normalizer": norm.to_dict(),
        "classes": classes,
        "patch_size": DEFAULT_PATCH_SIZE,
        "block_size": block_size,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "algorithms": {},
    }

    n_alg = len(algorithms)
    for i, algo in enumerate(algorithms):
        base, span = 0.12 + 0.8 * i / n_alg, 0.8 / n_alg
        progress(base, f"Training {ALGORITHM_LABELS[algo]}")
        if algo == "cnn":
            p_train, p_val, p_test = (extract_patches(feats_n, s, DEFAULT_PATCH_SIZE) for s in (train, val, test))
            model = CNNClassifier(len(names), classes, epochs=epochs, seed=seed)

            def cb(ep, total, rec, base=base, span=span):
                msg = f"CNN epoch {ep}/{total}  loss {rec['loss']:.3f}  train acc {rec['train_acc']:.3f}"
                if "val_acc" in rec:
                    msg += f"  val acc {rec['val_acc']:.3f}"
                progress(base + span * ep / total * 0.95, msg)

            model.fit(p_train, train.labels, p_val, val.labels, progress=cb)
            probs = model.predict_patches(p_test)
            model.save(out_dir / "cnn.pt")
            extra = {"history": model.history}
        else:
            model = ClassicalClassifier(algo, seed=seed).fit(x_train, train.labels, seed=seed)
            probs = model.predict_proba(x_test)
            model.save(out_dir / f"{algo}.joblib")
            imp = model.feature_importance()
            extra = {"feature_importance": dict(zip(names, imp))} if imp else {}
        pred = np.asarray(model.classes)[probs.argmax(1)]
        metrics = evaluate(test.labels, pred, CLASS_IDS)
        plot_confusion(metrics, out_dir / f"cm_{algo}.png", f"{ALGORITHM_LABELS[algo]} - test blocks")
        meta["algorithms"][algo] = {"label": ALGORITHM_LABELS[algo], "train_seconds": model.train_seconds,
                                    "test_metrics": metrics, **extra}
        progress(base + span, f"{ALGORITHM_LABELS[algo]}: test accuracy {metrics['overall_accuracy'] * 100:.1f}%")

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    progress(1.0, "Training complete")
    return meta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class ModelBundle:
    """A trained set of classifiers plus the feature normaliser they share."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.normalizer = Normalizer.from_dict(self.meta["normalizer"])
        self._cache: dict = {}

    @property
    def algorithms(self) -> list[str]:
        return list(self.meta["algorithms"].keys())

    def best_algorithm(self) -> str:
        return max(self.algorithms, key=lambda a: self.meta["algorithms"][a]["test_metrics"]["kappa"])

    def model(self, algo: str):
        if algo not in self.algorithms:
            raise ValueError(f"Algorithm '{algo}' not trained in bundle {self.path} (have {self.algorithms})")
        if algo not in self._cache:
            self._cache[algo] = (CNNClassifier.load(self.path / "cnn.pt") if algo == "cnn"
                                 else ClassicalClassifier.load(self.path / f"{algo}.joblib"))
        return self._cache[algo]

    def predict(self, scene: SceneData, algo: str) -> tuple[np.ndarray, np.ndarray]:
        """Returns (labels uint8 with CLOUD_CLASS for masked pixels, confidence float32)."""
        if self.meta["uses_dem"] and scene.dem is None:
            raise ValueError("This model was trained with DEM terrain features - please provide a DEM")
        dem = scene.dem if self.meta["uses_dem"] else None
        features, names = build_features(scene.reflectance, dem, scene.pixel_size_m)
        if names != self.meta["feature_names"]:
            raise ValueError("Feature mismatch between scene and model")
        feats_n = self.normalizer.transform_image(features)
        model = self.model(algo)
        if algo != "cnn":
            # classical models only need valid pixels -> much faster on cloudy scenes
            probs = np.zeros((len(model.classes), *scene.valid.shape), np.float32)
            idx = np.nonzero(scene.valid)
            if len(idx[0]):
                probs[:, idx[0], idx[1]] = model.predict_proba(feats_n[:, idx[0], idx[1]].T).T
        else:
            probs = model.predict_image(feats_n)
        classes = np.asarray(model.classes, dtype=np.uint8)
        labels = classes[probs.argmax(0)]
        conf = probs.max(0)
        labels[~scene.valid] = CLOUD_CLASS
        conf[~scene.valid] = 0
        return labels, conf.astype(np.float32)


def classify_scene(model_dir, scene_path, out_dir, algo: Optional[str] = None, dem_path=None,
                   cloud_path=None, reference_path=None, window: Optional[Sequence[int]] = None,
                   smoothing: int = 3, progress: Progress = _noop) -> dict:
    """Classify a scene with a trained bundle and write all map products to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = ModelBundle(model_dir)
    algo = algo or bundle.best_algorithm()
    t0 = time.time()
    progress(0.05, "Loading and preprocessing imagery")
    scene = load_scene(scene_path, dem_path, cloud_path, bundle.meta["sensor"], bundle.meta["dos"], window)
    progress(0.25, f"Classifying with {ALGORITHM_LABELS[algo]}")
    labels, conf = bundle.predict(scene, algo)
    if smoothing and smoothing > 1:
        progress(0.7, "Post-processing (majority filter)")
        labels = majority_filter(labels, smoothing)

    progress(0.8, "Writing map products")
    info = scene.info
    write_raster(out_dir / "classified.tif", labels[None], info, nodata=0, colormap=class_colormap(),
                 descriptions=["lithology"])
    write_raster(out_dir / "confidence.tif", conf[None], info, descriptions=["confidence"])
    save_png(colorize(labels), out_dir / "classified.png")
    save_png(composite(scene.reflectance, (2, 1, 0), scene.valid), out_dir / "rgb.png")
    save_png(composite(scene.reflectance, (5, 3, 0), scene.valid), out_dir / "falsecolor.png")
    save_png((np.clip(conf, 0, 1) * 255).astype(np.uint8), out_dir / "confidence.png")
    if scene.dem is not None:
        save_png(hillshade_png(scene.dem, scene.pixel_size_m), out_dir / "hillshade.png")
    plot_map_with_legend(labels, out_dir / "map_figure.png", f"Lithological map ({ALGORITHM_LABELS[algo]})")

    result = {
        "model": bundle.meta["name"],
        "algorithm": algo,
        "algorithm_label": ALGORITHM_LABELS[algo],
        "scene": str(scene_path),
        "window": list(window) if window is not None else None,
        "width": info.width, "height": info.height,
        "crs": info.crs.to_string() if info.crs else None,
        "bounds_wgs84": info.wgs84_bounds(),
        "pixel_size": info.pixel_size,
        "smoothing": smoothing,
        "mean_confidence": float(conf[scene.valid].mean()) if scene.valid.any() else 0.0,
        "masked_percent": float(100 * (~scene.valid).mean()),
        "area_stats": area_statistics(labels, info.pixel_area_km2),
        "seconds": None,
        "validation": None,
    }
    if reference_path:
        progress(0.9, "Validating against reference geological map")
        ref = load_labels(reference_path, info)
        m = compare_maps(labels, ref)
        plot_confusion(m, out_dir / "validation_cm.png", "Validation vs reference map")
        agreement = np.zeros((*labels.shape, 4), np.uint8)
        ok = (ref > 0) & scene.valid
        agreement[ok & (ref == labels)] = (46, 160, 67, 255)
        agreement[ok & (ref != labels)] = (215, 48, 39, 255)
        save_png(agreement, out_dir / "agreement.png")
        save_png(colorize(ref), out_dir / "reference.png")
        result["validation"] = m
    result["seconds"] = round(time.time() - t0, 2)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    progress(1.0, "Done")
    return result
