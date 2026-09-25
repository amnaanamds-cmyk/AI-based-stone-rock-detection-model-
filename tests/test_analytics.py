import json

import numpy as np
import pytest
from rasterio.transform import from_origin

from rockmap.analytics import (HAZARD_CLASSES, RobustStats, alteration_indices, alteration_score, find_targets,
                               fit_clusters, insights, landslide_susceptibility, predict_clusters,
                               spectral_features)
from rockmap.io import read_raster
from rockmap.region import Region, RegionConfig

ROCK = np.array([0.12, 0.15, 0.18, 0.22, 0.30, 0.28], np.float32)
GOSSAN = np.array([0.08, 0.12, 0.26, 0.30, 0.34, 0.317], np.float32)  # strong red / blue, rock-like SWIR
CLAY = np.array([0.20, 0.23, 0.26, 0.30, 0.42, 0.24], np.float32)     # Al-OH dip in SWIR2


def _scene(n=60):
    rng = np.random.default_rng(0)
    refl = np.broadcast_to(ROCK[:, None, None], (6, n, n)).copy() * (1 + 0.03 * rng.standard_normal((6, n, n)))
    refl[:, 10:20, 10:20] = GOSSAN[:, None, None]
    refl[:, 35:47, 30:44] = CLAY[:, None, None]
    return refl.astype(np.float32)


def test_alteration_targets_found_and_typed():
    refl = _scene()
    usable = np.ones(refl.shape[1:], bool)
    idx = alteration_indices(refl)
    stats = RobustStats.fit({k: v.ravel() for k, v in idx.items()})
    score, dom = alteration_score(refl, usable, stats)
    assert score[15, 15] > 75 and score[40, 36] > 75 and score[55, 5] < 30
    targets = find_targets(score, dom, from_origin(0, 1200, 20, 20), min_pixels=20)
    kinds = sorted(t["type"] for t in targets)
    assert kinds == ["clay", "iron_oxide"]
    assert all(t["area_ha"] > 0 and t["mean_score"] >= 75 for t in targets)


def test_alteration_ignores_snow_dark_and_vegetation():
    refl = _scene()
    refl[:, 10:20, 10:20] = np.array([0.03, 0.04, 0.05, 0.06, 0.06, 0.03])[:, None, None]      # dark
    refl[:, 35:47, 30:44] = np.array([0.03, 0.08, 0.05, 0.40, 0.22, 0.10])[:, None, None]      # vegetation
    idx = alteration_indices(refl)
    stats = RobustStats.fit({k: v.ravel() for k, v in idx.items()})
    score, _ = alteration_score(refl, np.ones(refl.shape[1:], bool), stats)
    assert score[15, 15] == 0 and score[40, 36] < 40


def test_landslide_susceptibility_increases_with_slope_and_rivers():
    n = 80
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    dem = np.where(xx < 40, 2000 + xx * 1.0, 2000 + (xx - 40) * 25.0)       # flat west, ~51 deg east
    water = np.zeros((n, n), bool)
    ndvi = np.zeros((n, n), np.float32)
    idx, cls = landslide_susceptibility(dem, 20.0, water, ndvi)
    assert cls[:, 5:30].mean() < cls[:, 50:75].mean()
    assert set(np.unique(cls)) <= set(HAZARD_CLASSES)
    water[:, 60] = True
    idx2, _ = landslide_susceptibility(dem, 20.0, water, ndvi, np.full((n, n), 7, np.uint8))
    assert idx2[:, 55:65].mean() > idx[:, 55:65].mean()


def test_clusters_separate_materials():
    refl = _scene()
    feats = spectral_features(refl)
    usable = np.ones(refl.shape[1:], bool)
    x = feats.reshape(feats.shape[0], -1).T
    mean, std, km = fit_clusters(x, 3)
    lab, conf = predict_clusters(feats, usable, mean, std, km)
    assert len({lab[15, 15], lab[40, 36], lab[55, 5]}) == 3
    assert 0 <= conf.min() and conf.max() <= 1


def test_insights_text():
    stats = {"classified_km2": 10, "districts": [], "region": [
        {"id": 4, "name": "Granite / Felsic Igneous", "pixels": 10, "area_km2": 6.0, "percent": 60.0},
        {"id": 7, "name": "Alluvium", "pixels": 5, "area_km2": 3.0, "percent": 30.0},
        {"id": 1, "name": "Limestone", "pixels": 2, "area_km2": 1.0, "percent": 10.0},
        {"id": 250, "name": "Snow", "pixels": 3, "area_km2": 2.0, "percent": None}]}
    analytics = {"targets_total": 3, "top_target": {"type_label": "Iron oxide", "mean_score": 91, "lat": 35.9, "lon": 74.3},
                 "hazard_km2": {"1": 5, "4": 2, "5": 1}, "clusters": [{"class_id": 4}, {"class_id": None}]}
    text = " ".join(insights(stats, analytics))
    assert "Granite" in text and "3 mineral-alteration anomalies" in text and "3 km²" in text and "Snow" in text


def test_region_analyze_units_and_field_validation(small_scene, tmp_path):
    _, info = read_raster(small_scene["scene"], bands=[1])
    w, s, e, n = info.wgs84_bounds()
    aoi = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    r = Region.create(tmp_path / "reg", RegionConfig(name="t", aoi=aoi, tile_size=64, source="local",
                                                     local_scenes=[str(small_scene["scene"])], local_sensor="reflectance",
                                                     local_dem=str(small_scene["dem"])))
    r.acquire(log=lambda m: None)
    res = r.analyze(6, log=lambda m: None)
    assert len(res["clusters"]) == 6 and sum(res["hazard_km2"].values()) > 1
    assert (r.folder / "products" / "targets.csv").exists()
    # name units by majority reference class -> a usable lithology map without training
    import rasterio
    from rockmap.pipeline import load_labels
    votes = {}
    for k in r.tile_keys:
        t = r.tile(k)
        with rasterio.open(r.tile_dir(k) / "analytics.tif") as src:
            cl = src.read(4)
        ref = load_labels(small_scene["reference"], t.info)
        for c in np.unique(cl[cl > 0]):
            v = np.bincount(ref[(cl == c) & (ref > 0)], minlength=8)
            votes.setdefault(int(c), np.zeros(8, int))
            votes[int(c)] += v
    mapping = {c: int(np.argmax(v)) for c, v in votes.items() if v.sum()}
    assert r.label_clusters(mapping)["classified"] == len(r.tile_keys)
    assert all(c["class_id"] for c in r.analytics()["clusters"] if c["id"] in mapping)
    paths = r.build_mosaics()
    assert {"alteration", "hazard", "clusters", "lithology"} <= set(paths)
    # field validation against the reference map's own labels at random points
    ref, rinfo = read_raster(small_scene["reference"])
    from rasterio.warp import transform
    obs = []
    for row, col in [(20, 20), (80, 40), (120, 130), (60, 100)]:
        x, y = rinfo.transform * (col + 0.5, row + 0.5)
        lo, la = transform(rinfo.crs, "EPSG:4326", [x], [y])
        obs.append({"lat": la[0], "lon": lo[0], "class_id": int(ref[0, row, col])})
    v = r.field_validation(obs)
    assert len(v["points"]) == 4 and v["n_compared"] >= 1
    from rockmap.report import region_report
    pdf = region_report(r, tmp_path / "r.pdf", None, r.statistics(), analytics=r.analytics(), validation=v)
    assert pdf.read_bytes()[:4] == b"%PDF"
