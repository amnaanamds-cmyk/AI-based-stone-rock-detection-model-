"""End-to-end pipeline: load data -> features -> train/benchmark -> classify -> validate -> map.

Everything is processed in blocks (with an overlap "halo" so CNN context, terrain
derivatives and the majority filter are seamless), so a full Sentinel-2 tile or a
region-wide mosaic never has to fit in memory.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from . import __version__
from .config import (ALGORITHM_LABELS, ALGORITHMS, CLASS_IDS, CLOUD_CLASS, DEFAULT_BLOCK_SIZE,
                     DEFAULT_PATCH_SIZE, DEFAULT_SAMPLES_PER_CLASS, MASK_BY_ID)
from .evaluation import confusion_matrix, evaluate, metrics_from_confusion
from .features import Normalizer, build_features
from .io import GeoInfo, align_to, class_colormap, read_raster, write_raster
from .mapping import (area_statistics_from_counts, colorize, composite, hillshade_png, majority_filter,
                      plot_confusion, plot_map_with_legend, save_png)
from .models import ClassicalClassifier, CNNClassifier
from .preprocessing import c_correction, illumination, landcover_mask, preprocess
from .sampling import block_split, extract_patches, pixel_vectors, sample_pixels

Progress = Callable[[float, str], None]


def _noop(_p: float, _m: str) -> None:
    pass


class Cancelled(Exception):
    """Raised by a progress callback to stop a long-running job."""


@dataclass
class SceneData:
    reflectance: np.ndarray    # (6, H, W)
    valid: np.ndarray          # (H, W) bool: finite, not cloud
    dem: Optional[np.ndarray]  # (H, W) metres
    info: GeoInfo
    landcover: Optional[np.ndarray] = None  # (H, W) uint8: 0 = rock, else mask class

    @property
    def pixel_size_m(self) -> float:
        px = self.info.pixel_size[0]
        if self.info.crs is not None and self.info.crs.is_geographic:
            return px * 111_320.0
        return px or 1.0

    @property
    def usable(self) -> np.ndarray:
        """Pixels where rock can be classified (valid and not snow/water/vegetation/shadow)."""
        return self.valid if self.landcover is None else self.valid & (self.landcover == 0)


@dataclass
class Sources:
    """Input layers of a scene and how to preprocess them."""
    scene: Path
    dem: Optional[Path] = None
    cloud: Optional[Path] = None
    sensor: str = "reflectance"
    dos: bool = False
    cloud_band: Optional[int] = None
    sun: Optional[tuple[float, float]] = None   # (azimuth, elevation) degrees
    topo_correct: bool = False
    masking: bool = True

    def shape(self) -> tuple[int, int]:
        with rasterio.open(self.scene) as src:
            return src.height, src.width

    def info(self) -> GeoInfo:
        with rasterio.open(self.scene) as src:
            return GeoInfo(src.transform, src.crs, src.width, src.height)


def parse_window(window: Optional[Sequence[int]]) -> Optional[Window]:
    """(col_off, row_off, width, height) -> rasterio Window."""
    if window is None:
        return None
    x, y, w, h = (int(v) for v in window)
    return Window(x, y, w, h)


def load_scene(scene_path, dem_path=None, cloud_path=None, sensor: str = "reflectance",
               dos: bool = False, window: Optional[Sequence[int] | Window] = None,
               cloud_band: Optional[int] = None, masking: bool = True,
               sun: Optional[tuple[float, float]] = None, topo_correct: bool = False) -> SceneData:
    """Read and preprocess a scene (and optional DEM / cloud layer) onto one grid."""
    win = window if isinstance(window, Window) else parse_window(window)
    with rasterio.open(scene_path) as src:
        full = Window(0, 0, src.width, src.height)
    if win is not None:
        win = win.intersection(full)
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
    scene = SceneData(refl, valid, dem, info)
    cos_i = None
    if dem is not None and sun is not None:
        cos_i = illumination(dem, scene.pixel_size_m, *sun)
        if topo_correct:
            scene.reflectance = c_correction(refl, cos_i, sun[1], valid)
    if masking:
        scl = cloud_layer if (cloud_layer is not None and sensor.startswith("sentinel2")) else None
        scene.landcover = landcover_mask(scene.reflectance, cos_i, scl)
        scene.landcover[~valid] = 0
    return scene


def load_sources(src: Sources, window: Optional[Window] = None) -> SceneData:
    return load_scene(src.scene, src.dem, src.cloud, src.sensor, src.dos, window, src.cloud_band,
                      src.masking, src.sun, src.topo_correct)


def load_labels(label_path, info: GeoInfo) -> np.ndarray:
    """Reference geological map (class ids) resampled onto the scene grid."""
    lab = align_to(label_path, info, Resampling.nearest)
    return np.nan_to_num(lab, nan=0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class SampleCollector:
    """Accumulates stratified train / validation / test samples from one or many tiles."""

    SPLITS = ("train", "val", "test")

    def __init__(self, per_class: int = DEFAULT_SAMPLES_PER_CLASS, block_size: int = DEFAULT_BLOCK_SIZE,
                 patches: bool = True, seed: int = 0, fractions: Sequence[float] = (0.6, 0.15, 0.25)):
        self.per_class = per_class
        self.block_size = block_size
        self.fractions = tuple(fractions)
        self.patches = patches
        self.seed = seed
        self.parts: dict[str, list] = {s: [] for s in self.SPLITS}
        self.n_sources = 0
        self.feature_names: Optional[list[str]] = None

    def caps(self, share: float = 1.0) -> dict[str, int]:
        base = {"train": self.per_class, "val": max(500, self.per_class // 4),
                "test": max(1000, self.per_class // 2)}
        return {k: max(1, int(math.ceil(v * share))) for k, v in base.items()}

    def add(self, features: np.ndarray, names: list[str], labels: np.ndarray, usable: np.ndarray,
            share: float = 1.0) -> int:
        """Sample from one scene/tile. ``share`` scales the per-class quota for multi-tile regions."""
        if self.feature_names is None:
            self.feature_names = list(names)
        elif list(names) != self.feature_names:
            raise ValueError("All tiles must provide the same features (DEM present everywhere or nowhere)")
        seed = self.seed + 7919 * self.n_sources
        split = block_split(labels.shape, self.block_size, self.fractions, seed=seed)
        margin = DEFAULT_PATCH_SIZE // 2
        caps = self.caps(share)
        added = 0
        for i, name in enumerate(self.SPLITS):
            s = sample_pixels(labels, usable, split, i, caps[name], margin, seed, erode_boundaries=name != "test")
            if not len(s):
                continue
            vec = pixel_vectors(features, s)
            pat = extract_patches(features, s, DEFAULT_PATCH_SIZE) if self.patches else None
            self.parts[name].append((vec, pat, s.labels))
            added += len(s)
        self.n_sources += 1
        return added

    def arrays(self, name: str) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        parts = self.parts[name]
        if not parts:
            return np.zeros((0, len(self.feature_names or []))), None, np.zeros(0, np.int64)
        vec = np.concatenate([p[0] for p in parts])
        pat = np.concatenate([p[1] for p in parts]) if self.patches else None
        lab = np.concatenate([p[2] for p in parts])
        # enforce the global per-class cap after pooling tiles
        cap = self.caps()[name]
        rng = np.random.default_rng(self.seed + self.SPLITS.index(name))
        keep = []
        for c in np.unique(lab):
            idx = np.nonzero(lab == c)[0]
            keep.append(rng.choice(idx, size=min(cap, len(idx)), replace=False))
        keep = rng.permutation(np.concatenate(keep))
        return vec[keep], (pat[keep] if pat is not None else None), lab[keep]


def fit_bundle(collector: SampleCollector, out_dir, algorithms: Sequence[str], meta: dict,
               epochs: int = 30, seed: int = 0, progress: Progress = _noop,
               base: float = 0.1, span: float = 0.9, cnn_options: Optional[dict] = None) -> dict:
    """Fit the normaliser and every requested classifier on collected samples, save the bundle.

    ``cnn_options`` may set ``batch_size``, ``lr`` and ``width`` of the CNN."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_train, p_train, y_train = collector.arrays("train")
    x_val, p_val, y_val = collector.arrays("val")
    x_test, p_test, y_test = collector.arrays("test")
    if len(y_train) < 50:
        raise ValueError("Too few labelled training pixels - check that the reference map overlaps the "
                         "imagery and that the labelled areas are not masked (snow, cloud, vegetation...)")
    if len(y_test) == 0:
        x_test, p_test, y_test = x_val, p_val, y_val
    classes = sorted(int(c) for c in np.unique(y_train))
    names = collector.feature_names
    norm = Normalizer().fit(x_train)
    x_train, x_val, x_test = (norm.transform_pixels(x) for x in (x_train, x_val, x_test))
    if collector.patches:
        m, s = norm.mean[None, :, None, None], norm.std[None, :, None, None]
        p_train, p_val, p_test = ((p - m) / s if p is not None else None for p in (p_train, p_val, p_test))

    meta = {**meta, "version": __version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "feature_names": names, "normalizer": norm.to_dict(), "classes": classes,
            "patch_size": DEFAULT_PATCH_SIZE, "block_size": collector.block_size,
            "n_train": int(len(y_train)), "n_val": int(len(y_val)), "n_test": int(len(y_test)),
            "class_counts_train": {int(c): int((y_train == c).sum()) for c in classes},
            "algorithms": {}}
    n_alg = len(algorithms)
    for i, algo in enumerate(algorithms):
        b, sp = base + span * i / n_alg, span / n_alg
        progress(b, f"Training {ALGORITHM_LABELS[algo]}")
        if algo == "cnn":
            if not collector.patches:
                raise ValueError("CNN requested but patches were not collected")
            model = CNNClassifier(len(names), classes, epochs=epochs, seed=seed, **(cnn_options or {}))

            def cb(ep, total, rec, b=b, sp=sp):
                msg = f"CNN epoch {ep}/{total}  loss {rec['loss']:.3f}  train acc {rec['train_acc']:.3f}"
                if "val_acc" in rec:
                    msg += f"  val acc {rec['val_acc']:.3f}"
                progress(b + sp * ep / total * 0.95, msg)

            model.fit(p_train.astype(np.float32), y_train, p_val.astype(np.float32) if len(y_val) else None,
                      y_val, progress=cb)
            probs = model.predict_patches(p_test.astype(np.float32))
            model.save(out_dir / "cnn.pt")
            extra = {"history": model.history}
        else:
            model = ClassicalClassifier(algo, seed=seed).fit(x_train, y_train, seed=seed)
            probs = model.predict_proba(x_test)
            model.save(out_dir / f"{algo}.joblib")
            imp = model.feature_importance()
            extra = {"feature_importance": dict(zip(names, imp))} if imp else {}
        pred = np.asarray(model.classes)[probs.argmax(1)]
        metrics = evaluate(y_test, pred, CLASS_IDS)
        plot_confusion(metrics, out_dir / f"cm_{algo}.png", f"{ALGORITHM_LABELS[algo]} - test blocks")
        meta["algorithms"][algo] = {"label": ALGORITHM_LABELS[algo], "train_seconds": model.train_seconds,
                                    "test_metrics": metrics, **extra}
        progress(b + sp, f"{ALGORITHM_LABELS[algo]}: test accuracy {metrics['overall_accuracy'] * 100:.1f}%")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def train_models(scene_path, label_path, out_dir, dem_path=None, cloud_path=None,
                 algorithms: Sequence[str] = ALGORITHMS, sensor: str = "reflectance", dos: bool = False,
                 samples_per_class: int = DEFAULT_SAMPLES_PER_CLASS, epochs: int = 30,
                 block_size: int = DEFAULT_BLOCK_SIZE, seed: int = 0, name: Optional[str] = None,
                 progress: Progress = _noop, masking: bool = True,
                 sun: Optional[tuple[float, float]] = None, topo_correct: bool = False) -> dict:
    """Train the CNN and benchmark classifiers on one scene; write a model bundle to ``out_dir``.

    Bundle layout::

        meta.json        feature names, normaliser, classes, metrics for every algorithm
        cnn.pt           PyTorch CNN weights
        rf.joblib        Random Forest
        svm.joblib       Support Vector Machine
        cm_<algo>.png    confusion matrices on the held-out spatial test blocks
    """
    algorithms = [a for a in algorithms if a in ALGORITHMS]
    if not algorithms:
        raise ValueError(f"No valid algorithm given (choose from {', '.join(ALGORITHMS)})")
    progress(0.02, "Loading and preprocessing imagery")
    scene = load_scene(scene_path, dem_path, cloud_path, sensor, dos, masking=masking, sun=sun,
                       topo_correct=topo_correct)
    labels = load_labels(label_path, scene.info)
    features, names = build_features(scene.reflectance, scene.dem, scene.pixel_size_m)
    progress(0.08, "Preparing training samples (spatial block split)")
    col = SampleCollector(samples_per_class, block_size, patches="cnn" in algorithms, seed=seed)
    col.add(features, names, labels, scene.usable)
    meta = {"name": name or Path(out_dir).name, "scene": str(scene_path), "labels": str(label_path),
            "sensor": sensor, "dos": dos, "uses_dem": scene.dem is not None, "masking": masking,
            "topo_correct": topo_correct}
    meta = fit_bundle(col, out_dir, algorithms, meta, epochs, seed, progress, 0.12, 0.86)
    progress(1.0, "Training complete")
    return meta


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class ModelBundle:
    """A trained set of classifiers plus the feature normaliser they share."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text(encoding="utf-8"))
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
        """Returns (labels uint8, confidence float32).

        Masked pixels get CLOUD_CLASS (no data / cloud) or their land-cover class
        (snow, water, vegetation, shadow).
        """
        if self.meta["uses_dem"] and scene.dem is None:
            raise ValueError("This model was trained with DEM terrain features - please provide a DEM")
        dem = scene.dem if self.meta["uses_dem"] else None
        features, names = build_features(scene.reflectance, dem, scene.pixel_size_m)
        if names != self.meta["feature_names"]:
            raise ValueError("Feature mismatch between scene and model")
        model = self.model(algo)
        usable = scene.usable
        classes = np.asarray(model.classes, dtype=np.uint8)
        probs = np.zeros((len(classes), *usable.shape), np.float32)
        if algo != "cnn":
            # classical models only need usable pixels -> much faster on cloudy / snowy scenes
            idx = np.nonzero(usable)
            if len(idx[0]):
                x = self.normalizer.transform_pixels(features[:, idx[0], idx[1]].T)
                probs[:, idx[0], idx[1]] = model.predict_proba(x).T
        elif usable.any():
            probs = model.predict_image(self.normalizer.transform_image(features))
        labels = classes[probs.argmax(0)]
        conf = probs.max(0)
        if scene.landcover is not None:
            lc = scene.landcover > 0
            labels[lc] = scene.landcover[lc]
            conf[lc] = 0
        labels[~scene.valid] = CLOUD_CLASS
        conf[~scene.valid] = 0
        return labels, conf.astype(np.float32)


def iter_blocks(width: int, height: int, block: int) -> Iterator[Window]:
    for y in range(0, height, block):
        for x in range(0, width, block):
            yield Window(x, y, min(block, width - x), min(block, height - y))


def predict_window(bundle: ModelBundle, algo: str, sources: Sources, core: Window, halo: int,
                   full: tuple[int, int], smoothing: int = 3) -> tuple[np.ndarray, np.ndarray, SceneData, tuple]:
    """Classify ``core`` (scene pixel coords) reading ``halo`` extra pixels of context on each side.

    Returns (labels, confidence, scene data of the halo read, (row_off, col_off) of core within it).
    """
    h, w = full
    x0, y0 = max(0, int(core.col_off) - halo), max(0, int(core.row_off) - halo)
    x1 = min(w, int(core.col_off + core.width) + halo)
    y1 = min(h, int(core.row_off + core.height) + halo)
    scene = load_sources(sources, Window(x0, y0, x1 - x0, y1 - y0))
    labels, conf = bundle.predict(scene, algo)
    if smoothing and smoothing > 1:
        labels = majority_filter(labels, smoothing)
    oy, ox = int(core.row_off) - y0, int(core.col_off) - x0
    sl = (slice(oy, oy + int(core.height)), slice(ox, ox + int(core.width)))
    return labels[sl], conf[sl], scene, sl


def _tiled_profile(info: GeoInfo, count: int, dtype: str, nodata=None) -> dict:
    prof = dict(driver="GTiff", width=info.width, height=info.height, count=count, dtype=dtype,
                transform=info.transform, compress="deflate", tiled=True, blockxsize=256, blockysize=256,
                BIGTIFF="IF_SAFER")
    if info.crs is not None:
        prof["crs"] = info.crs
    if nodata is not None:
        prof["nodata"] = nodata
    return prof


def read_preview(path, max_size: int = 2048, resampling=Resampling.nearest, bands=None) -> np.ndarray:
    with rasterio.open(path) as src:
        scale = max(1.0, max(src.width, src.height) / max_size)
        shape = (max(1, int(round(src.height / scale))), max(1, int(round(src.width / scale))))
        idx = bands or list(range(1, src.count + 1))
        return src.read(idx, out_shape=(len(idx), *shape), resampling=resampling)


@dataclass
class ClassifyAccumulator:
    counts: np.ndarray = field(default_factory=lambda: np.zeros(256, np.int64))
    conf_sum: float = 0.0
    conf_n: int = 0
    cm: Optional[np.ndarray] = None

    def add(self, labels: np.ndarray, conf: np.ndarray, ref: Optional[np.ndarray] = None):
        self.counts += np.bincount(labels.ravel(), minlength=256)
        rock = np.isin(labels, CLASS_IDS)
        self.conf_sum += float(conf[rock].sum())
        self.conf_n += int(rock.sum())
        if ref is not None:
            ok = (ref > 0) & rock
            cm = confusion_matrix(ref[ok], labels[ok], CLASS_IDS)
            self.cm = cm if self.cm is None else self.cm + cm


def classify_scene(model_dir, scene_path, out_dir, algo: Optional[str] = None, dem_path=None,
                   cloud_path=None, reference_path=None, window: Optional[Sequence[int]] = None,
                   smoothing: int = 3, progress: Progress = _noop, block: int = 1024, masking: bool = True,
                   sun: Optional[tuple[float, float]] = None, sensor: Optional[str] = None) -> dict:
    """Classify a scene (or a window of it) with a trained bundle; write all map products."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = ModelBundle(model_dir)
    algo = algo or bundle.best_algorithm()
    t0 = time.time()
    sources = Sources(Path(scene_path), Path(dem_path) if dem_path else None,
                      Path(cloud_path) if cloud_path else None, sensor or bundle.meta["sensor"],
                      bundle.meta["dos"], sun=sun, topo_correct=bundle.meta.get("topo_correct", False),
                      masking=masking)
    full_info = sources.info()
    H, W = full_info.height, full_info.width
    aoi = parse_window(window) or Window(0, 0, W, H)
    aoi = aoi.intersection(Window(0, 0, W, H))
    info = full_info.window(aoi)
    halo = DEFAULT_PATCH_SIZE // 2 + max(1, smoothing) + 2

    lith_path, conf_path = out_dir / "classified.tif", out_dir / "confidence.tif"
    agree_path = out_dir / "agreement.tif"
    acc = ClassifyAccumulator()
    blocks = list(iter_blocks(info.width, info.height, block))
    with rasterio.open(lith_path, "w", **_tiled_profile(info, 1, "uint8", 0)) as dl, \
            rasterio.open(conf_path, "w", **_tiled_profile(info, 1, "uint8")) as dc:
        dl.write_colormap(1, class_colormap())
        dl.set_band_description(1, "lithology")
        dc.set_band_description(1, "confidence (%)")
        da = rasterio.open(agree_path, "w", **_tiled_profile(info, 1, "uint8", 0)) if reference_path else None
        try:
            for i, b in enumerate(blocks):
                progress(0.05 + 0.75 * i / len(blocks),
                         f"Classifying block {i + 1}/{len(blocks)} with {ALGORITHM_LABELS[algo]}")
                core = Window(aoi.col_off + b.col_off, aoi.row_off + b.row_off, b.width, b.height)
                labels, conf, scene, sl = predict_window(bundle, algo, sources, core, halo, (H, W), smoothing)
                dl.write(labels[None], window=b)
                dc.write(np.round(conf * 100).astype(np.uint8)[None], window=b)
                ref = None
                if reference_path:
                    ref = load_labels(reference_path, full_info.window(core))
                    agree = np.zeros(ref.shape, np.uint8)
                    ok = (ref > 0) & np.isin(labels, CLASS_IDS)
                    agree[ok & (ref == labels)] = 1
                    agree[ok & (ref != labels)] = 2
                    da.write(agree[None], window=b)
                acc.add(labels, conf, ref)
        finally:
            if da is not None:
                da.write_colormap(1, {0: (0, 0, 0, 0), 1: (46, 160, 67, 255), 2: (215, 48, 39, 255)})
                da.close()

    progress(0.82, "Rendering previews and map figure")
    lab_prev = read_preview(lith_path)[0]
    save_png(colorize(lab_prev), out_dir / "classified.png")
    save_png(read_preview(conf_path)[0], out_dir / "confidence.png")
    _scene_previews(sources, aoi, full_info, out_dir)
    plot_map_with_legend(lab_prev, out_dir / "map_figure.png", f"Lithological map ({ALGORITHM_LABELS[algo]})")

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
        "mean_confidence": acc.conf_sum / acc.conf_n if acc.conf_n else 0.0,
        "masked_percent": float(100 * sum(acc.counts[c] for c in MASK_BY_ID) / max(1, acc.counts.sum())),
        "area_stats": area_statistics_from_counts(acc.counts, info.pixel_area_km2),
        "seconds": None,
        "validation": None,
    }
    if reference_path:
        progress(0.9, "Validating against reference geological map")
        m = metrics_from_confusion(acc.cm if acc.cm is not None else np.zeros((7, 7), int), CLASS_IDS)
        plot_confusion(m, out_dir / "validation_cm.png", "Validation vs reference map")
        ag = read_preview(agree_path)[0]
        rgba = np.zeros((*ag.shape, 4), np.uint8)
        rgba[ag == 1] = (46, 160, 67, 255)
        rgba[ag == 2] = (215, 48, 39, 255)
        save_png(rgba, out_dir / "agreement.png")
        ref_prev = load_labels(reference_path, _preview_info(info, lab_prev.shape))
        save_png(colorize(ref_prev), out_dir / "reference.png")
        result["validation"] = m
    result["seconds"] = round(time.time() - t0, 2)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    progress(1.0, "Done")
    return result


def _preview_info(info: GeoInfo, shape: tuple[int, int]) -> GeoInfo:
    sy, sx = info.height / shape[0], info.width / shape[1]
    return GeoInfo(info.transform @ info.transform.scale(sx, sy), info.crs, shape[1], shape[0])


def _scene_previews(sources: Sources, aoi: Window, full_info: GeoInfo, out_dir: Path, max_size: int = 2048):
    """True colour, SWIR false colour and hillshade previews of the classified area."""
    from .preprocessing import to_reflectance
    info = full_info.window(aoi)
    scale = max(1.0, max(info.width, info.height) / max_size)
    shape = (max(1, int(round(info.height / scale))), max(1, int(round(info.width / scale))))
    with rasterio.open(sources.scene) as src:
        raw = src.read(list(range(1, 7)), window=aoi, out_shape=(6, *shape),
                       resampling=Resampling.average).astype(np.float32)
        if src.nodata is not None:
            raw[raw == src.nodata] = np.nan
    refl = to_reflectance(raw, sources.sensor) if sources.sensor != "reflectance" else raw
    valid = np.all(np.isfinite(refl), 0) & (np.nansum(refl, 0) > 0)
    save_png(composite(refl, (2, 1, 0), valid), out_dir / "rgb.png")
    save_png(composite(refl, (5, 3, 0), valid), out_dir / "falsecolor.png")
    if sources.dem is not None:
        pinfo = _preview_info(info, shape)
        dem = align_to(sources.dem, pinfo)
        dem = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0)
        save_png(hillshade_png(dem, pinfo.pixel_size[0] or 1.0), out_dir / "hillshade.png")
