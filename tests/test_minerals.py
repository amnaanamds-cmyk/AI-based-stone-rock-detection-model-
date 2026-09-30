import json

import numpy as np
import pytest
from rasterio.transform import from_origin
from scipy import ndimage

from rockmap.io import read_raster
from rockmap.minerals import (lineament_density, lineament_segments, lineaments, parse_occurrences, quartz_index,
                              rose)
from rockmap.region import Region, RegionConfig


def _fault_dem(n=300):
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:n, 0:n]
    dem = 3000 + ndimage.gaussian_filter(rng.normal(0, 1, (n, n)), 8) * 300
    dem -= 150 * np.exp(-(((xx - (n - yy)) / np.sqrt(2)) / 3) ** 2)      # straight NE-SW fault valley
    return dem.astype(np.float32)


def test_lineaments_find_the_fault_and_its_strike():
    dem = _fault_dem()
    mask, strike = lineaments(dem, 20.0)
    segs = lineament_segments(mask, strike, from_origin(0, 6000, 20, 20), 20.0)
    main = max(rose(segs), key=lambda r: r["km"])
    assert 30 <= main["from"] <= 60                                    # NE strike dominates
    assert any(s["length_m"] >= 2000 and 30 <= s["strike"] <= 60 for s in segs)
    d = lineament_density(mask, 20.0)
    assert d[150, 150] > np.median(d)                                  # denser along the fault


def test_quartz_index_rises_with_quartz():
    quartzite = np.array([0.75, 0.86, 0.74])   # reststrahlen: bands 10 and 12 in minima[:, None, None] * np.ones((3, 2, 2))
    basalt = np.array([0.95, 0.95, 0.95])[:, None, None] * np.ones((3, 2, 2))
    assert quartz_index(quartzite).mean() > quartz_index(basalt).mean()


def test_parse_occurrences():
    occ = parse_occurrences(b"lat,lon,commodity,name\n35.9,74.3,Copper,A\n35.8,74.2,stibnite,B\n35,74,lithium,C\n")
    assert [o["model"] for o in occ] == ["copper", "vein", None]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"commodity": "haematite"},
                                                     "geometry": {"type": "Point", "coordinates": [74.1, 35.1]}}]}
    assert parse_occurrences(json.dumps(gj).encode(), "x.geojson")[0]["model"] == "iron"
    with pytest.raises(ValueError):
        parse_occurrences(b"lat,lon,commodity\n100,74,iron\n")


def test_region_mineral_analysis(small_scene, tmp_path):
    ref, rinfo = read_raster(small_scene["reference"])
    w, s, e, n = rinfo.wgs84_bounds()
    aoi = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    r = Region.create(tmp_path / "reg", RegionConfig(name="m", aoi=aoi, tile_size=64, source="local",
                                                     local_scenes=[str(small_scene["scene"])],
                                                     local_sensor="reflectance", local_dem=str(small_scene["dem"])))
    r.acquire(log=lambda m: None)
    from rasterio.warp import transform
    rng = np.random.default_rng(0)
    rr, cc = np.nonzero(np.isfinite(ref[0]))
    pick = rng.choice(len(rr), 10, replace=False)
    xs, ys = rinfo.transform * (cc[pick] + 0.5, rr[pick] + 0.5)
    lons, lats = transform(rinfo.crs, "EPSG:4326", list(xs), list(ys))
    occ = [{"lat": la, "lon": lo, "commodity": "iron", "model": "iron", "name": f"O{i}"}
           for i, (lo, la) in enumerate(zip(lons, lats))]
    res = r.mineral_analysis(occ, log=lambda m: None)
    assert {m["key"] for m in res["models"]} == {"iron", "copper", "vein"}
    assert res["lineaments"]["segments"] > 0 and len(res["lineaments"]["rose"]) == 12
    assert res["occurrences_inside"] >= 8 and res["ml"]["occurrences"] >= 8
    for f in ("mineral_targets.csv", "mineral_targets.geojson", "lineaments.geojson", "minerals.json"):
        assert (r.folder / "products" / f).exists()
    lines = json.loads((r.folder / "products" / "lineaments.geojson").read_text())
    assert lines["features"][0]["geometry"]["type"] == "LineString"
    paths = r.build_mosaics()
    assert {"minerals", "min_iron", "min_copper", "min_vein", "min_ml", "lineaments"} <= set(paths)
    from rockmap.report import region_report
    pdf = region_report(r, tmp_path / "m.pdf", None, r.statistics(), minerals=r.minerals())
    assert pdf.read_bytes()[:4] == b"%PDF"
