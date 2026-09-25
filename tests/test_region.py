import json

import numpy as np
import rasterio

from rockmap.io import read_raster
from rockmap.presets import GILGIT_BALTISTAN, PRESETS
from rockmap.region import Region, RegionConfig


def test_gilgit_baltistan_grid(tmp_path):
    r = Region.create(tmp_path / "gb", RegionConfig(name="GB", aoi=GILGIT_BALTISTAN))
    assert r.crs.to_epsg() == 32643
    assert 150 < len(r.tile_keys) < 260
    assert r.config.tile_size * r.config.resolution == 20480
    assert set(PRESETS) >= {"gilgit-baltistan", "gilgit", "hunza", "skardu"}


def test_region_end_to_end_local(small_scene, tmp_path):
    _, info = read_raster(small_scene["scene"], bands=[1])
    w, s, e, n = info.wgs84_bounds()
    aoi = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    cfg = RegionConfig(name="t", aoi=aoi, tile_size=64, source="local", local_scenes=[str(small_scene["scene"])],
                       local_sensor="reflectance", local_dem=str(small_scene["dem"]))
    r = Region.create(tmp_path / "reg", cfg)
    assert r.acquire(log=lambda m: None)["acquired"] == len(r.tile_keys) > 4
    meta = r.train(tmp_path / "model", label_rasters=[small_scene["reference"]], algorithms=["rf", "cnn"],
                   samples_per_class=300, epochs=2, log=lambda m: None)
    assert meta["algorithms"]["rf"]["test_metrics"]["overall_accuracy"] > 0.7
    assert r.classify(tmp_path / "model", "rf", log=lambda m: None)["classified"] == len(r.tile_keys)
    paths = r.build_mosaics()
    with rasterio.open(paths["lithology"]) as src:
        lab = src.read(1)
        assert src.width == r.width and src.height == r.height
    # seamless: the mosaic agrees with the reference map across tile edges too
    from rockmap.pipeline import load_labels
    from rockmap.io import GeoInfo
    ref = load_labels(small_scene["reference"], GeoInfo(src.transform, src.crs, src.width, src.height))
    ok = (ref > 0) & (lab > 0) & (lab < 250)
    assert (ref[ok] == lab[ok]).mean() > 0.75
    st = r.statistics()
    total = sum(x["area_km2"] for x in st["region"])
    assert abs(total - (160 * 0.02) ** 2) < 0.5          # 160 x 160 px of 20 m
    q = r.query((w + e) / 2, (s + n) / 2)
    assert q["inside"] and q["class_id"] > 0 and "elevation_m" in q
    out = r.export_geojson(tmp_path / "poly.geojson", min_pixels=10)
    gj = json.loads(out.read_text())
    assert gj["features"] and {"class_id", "lithology", "area_ha", "tile"} <= set(gj["features"][0]["properties"])
    from rockmap.report import region_report
    pdf = region_report(r, tmp_path / "r.pdf", meta, st)
    assert pdf.read_bytes()[:4] == b"%PDF"


def test_region_resume_skips_done_tiles(small_scene, tmp_path):
    _, info = read_raster(small_scene["scene"], bands=[1])
    w, s, e, n = info.wgs84_bounds()
    aoi = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    r = Region.create(tmp_path / "reg", RegionConfig(name="t", aoi=aoi, tile_size=128, source="local",
                                                     local_scenes=[str(small_scene["scene"])],
                                                     local_sensor="reflectance"))
    first = r.tile_keys[0]
    r.acquire(log=lambda m: None, keys=[first])
    mtime = (r.tile_dir(first) / "stack.tif").stat().st_mtime
    r.acquire(log=lambda m: None)
    assert (r.tile_dir(first) / "stack.tif").stat().st_mtime == mtime
    assert r.summary()["acquired"] == len(r.tile_keys)
