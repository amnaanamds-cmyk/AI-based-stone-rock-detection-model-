"""`rockmap quickstart`: build a complete, ready-to-present demo in the dashboard.

Offline demo (always): a synthetic mountain valley processed end to end as a *region*
(imagery → analytics → CNN/RF/SVM training from a digitised reference map → classification →
mosaics, statistics per district, targets, GeoJSON, PDF), plus field observations and a public
share link. With ``--real``: also a real 40 x 40 km area around Gilgit city downloaded from Sentinel-2
and the Copernicus DEM (analytics only, since no reference geology is bundled).
"""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Callable

import numpy as np

DEMO_NAME = "Demo: synthetic Karakoram valley"
GB_NAME = "Gilgit-Baltistan overview (real Sentinel-2, 100 m)"
REAL_NAME = "Gilgit & surroundings (real Sentinel-2, 40 x 40 km)"
DEMO_PASSWORD = "rockmap-demo"


def _noop(_p, _m):
    pass


def _new_region(db, root: Path, name: str, cfg, preset=None) -> int:
    from .region import Region, _geom_bounds
    rid = db.insert("regions", name=name, folder="", preset=preset, created_by="admin")
    folder = root / "regions" / str(rid)
    Region.create(folder, cfg)
    db.update("regions", rid, folder=folder.relative_to(root).as_posix(), bounds=list(_geom_bounds(cfg.aoi)),
              share_token=secrets.token_urlsafe(18))
    return rid


def _run(db, root: Path, kind: str, params: dict, **cols) -> dict:
    from .web.jobs import _claim, enqueue, run_job
    import time
    job_id = enqueue(db, kind, params, user="admin", **cols)
    # run it here, unless a running RockMap server's worker picked it up first - then wait for it
    while True:
        job = _claim(db, "quickstart")
        if job:
            run_job(db, root, job)
            continue
        job = db.get("jobs", job_id)
        if job["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(2)
    if job["status"] != "done":
        raise RuntimeError(f"{kind} failed: {job['error']}")
    return job


def build_demo(root: Path, real: bool = False, progress: Callable = _noop, size: int = 512, epochs: int = 12,
               gb: bool = False) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fresh = not (root / "rockmap.db").exists()
    if not os.environ.get("ROCKMAP_ADMIN_PASSWORD") and fresh:
        os.environ["ROCKMAP_ADMIN_PASSWORD"] = DEMO_PASSWORD
    from .web.app import create_app
    app = create_app(root, sync_jobs=True, workers=0)
    db = app.extensions["rockmap_db"]
    if not db.one("regions", "name = ?", (DEMO_NAME,)):
        _synthetic_region(db, root, progress, size, epochs)
    else:
        progress(1.0, "Demo region already present")
    if real and not db.one("regions", "name = ?", (REAL_NAME,)):
        _real_region(db, root, progress)
    if gb and not db.one("regions", "name = ?", (GB_NAME,)):
        _gb_overview(db, root, progress)
    pw = os.environ.get("ROCKMAP_ADMIN_PASSWORD") if fresh else None
    print("\n" + "=" * 68)
    print(" RockMap demo ready.  Sign in as  admin  /  " + (pw or f"your existing password (demo default: {DEMO_PASSWORD})"))
    print(" Change the password under Profile before sharing the server.")
    for r in db.all("regions"):
        if r.get("share_token"):
            print(f" Public share link for '{r['name']}':  /share/{r['share_token']}")
    print("=" * 68 + "\n", flush=True)


def _synthetic_region(db, root: Path, progress: Callable, size: int = 512, epochs: int = 12) -> None:
    from rasterio.features import shapes
    from rasterio.warp import transform_geom

    from .io import read_raster
    from .region import RegionConfig
    from .synthetic import write_scene
    progress(0.02, "Generating synthetic study area")
    src = write_scene(root / "demo_source", size, size, seed=11, cloud_cover=0.02)
    _, info = read_raster(src["scene"], bands=[1])
    w, s, e, n = info.wgs84_bounds()
    aoi = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    cfg = RegionConfig(name=DEMO_NAME, aoi=aoi, tile_size=max(128, size // 2), source="local", local_scenes=[str(src["scene"])],
                       local_sensor="reflectance", local_dem=str(src["dem"]))
    rid = _new_region(db, root, DEMO_NAME, cfg, "demo")
    folder = root / "regions" / str(rid)

    # "digitised geological map": polygons of the western 60 % of the reference map
    ref, rinfo = read_raster(src["reference"])
    lab = np.nan_to_num(ref[0]).astype(np.uint8)
    lab[:, int(lab.shape[1] * 0.6):] = 0
    feats = []
    for geom, val in shapes(lab, mask=lab > 0, transform=rinfo.transform):
        if len(geom["coordinates"][0]) >= 6:
            feats.append({"type": "Feature", "geometry": transform_geom(rinfo.crs, "EPSG:4326", geom, precision=6),
                          "properties": {"UNIT": int(val)}})
    (folder / "references").mkdir(exist_ok=True)
    (folder / "references" / "ref1_demo_geology.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    (folder / "references" / "index.json").write_text(json.dumps([{
        "file": "ref1_demo_geology.geojson", "name": "demo_geology.geojson", "field": "UNIT", "mapping": None,
        "polygons": len(feats), "by": "admin"}]), encoding="utf-8")

    # district boundaries (two halves) for per-district statistics
    mid = (s + n) / 2
    districts = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"name": "North district"},
         "geometry": {"type": "Polygon", "coordinates": [[[w, mid], [e, mid], [e, n], [w, n], [w, mid]]]}},
        {"type": "Feature", "properties": {"name": "South district"},
         "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, mid], [w, mid], [w, s]]]}}]}
    (folder / "districts.geojson").write_text(json.dumps(districts), encoding="utf-8")
    (folder / "districts.json").write_text(json.dumps({"field": "name"}), encoding="utf-8")

    # field observations in the eastern part (not covered by the reference map) for validation
    from rasterio.warp import transform
    truth = np.nan_to_num(ref[0]).astype(np.uint8)
    rng = np.random.default_rng(3)
    rr, cc = np.nonzero(truth > 0)
    east = cc > truth.shape[1] * 0.65
    pick = rng.choice(np.nonzero(east)[0], 30, replace=False)
    xs, ys = rinfo.transform * (cc[pick] + 0.5, rr[pick] + 0.5)
    lons, lats = transform(rinfo.crs, "EPSG:4326", list(xs), list(ys))
    for lo, la, p in zip(lons, lats, pick):
        db.insert("observations", region_id=rid, lat=la, lon=lo, accuracy_m=5.0, class_id=int(truth[rr[p], cc[p]]),
                  certainty=1, note="demo field check (not used for training)", author="demo-geologist")

    progress(0.1, "Running the full region pipeline (acquire → analytics → train → classify → products)")
    _run(db, root, "region_pipeline", {"algorithms": ["cnn", "rf", "svm"], "epochs": epochs, "samples": 2000,
                                       "smoothing": 3, "n_clusters": 8, "geojson": True, "min_pixels": 25,
                                       "name": "Demo valley model (CNN + RF + SVM)"}, region_id=rid)
    progress(0.95, "Demo region complete")


def _gb_overview(db, root: Path, progress: Callable) -> None:
    """Whole Gilgit-Baltistan at 100 m: imagery, rock units, alteration, hazard and gem layers (30-60 min)."""
    from .presets import PRESETS
    from .region import RegionConfig
    progress(0.0, "Creating the whole Gilgit-Baltistan overview (downloads Sentinel-2 + Copernicus DEM, 30-60 min)")
    p = PRESETS["gilgit-baltistan-overview"]
    cfg = RegionConfig(name=GB_NAME, aoi=p["geometry"], resolution=p["resolution"])
    rid = _new_region(db, root, GB_NAME, cfg, "gilgit-baltistan-overview")
    _run(db, root, "region_pipeline", {"n_clusters": 12, "geojson": False}, region_id=rid)
    progress(1.0, "Gilgit-Baltistan overview complete")


def _real_region(db, root: Path, progress: Callable) -> None:
    from .presets import PRESETS
    from .region import RegionConfig
    progress(0.0, "Creating real Gilgit region (downloads Sentinel-2 + Copernicus DEM)")
    p = PRESETS["gilgit"]
    cfg = RegionConfig(name=REAL_NAME, aoi=p["geometry"], years=[2023, 2024], max_scenes=4)
    rid = _new_region(db, root, REAL_NAME, cfg, "gilgit")
    _run(db, root, "region_pipeline", {"n_clusters": 10, "geojson": False}, region_id=rid)
    progress(1.0, "Real Gilgit region complete")
