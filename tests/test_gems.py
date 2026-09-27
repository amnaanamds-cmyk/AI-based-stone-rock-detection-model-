import json

import numpy as np
import pytest

from rockmap.gems import (GEM_MODELS, EVIDENCE, EvidenceStats, evidence_layers, model_scores, parse_occurrences,
                          raw_evidence, success_rates)
from rockmap.io import read_raster
from rockmap.region import Region, RegionConfig

HOST = np.array([0.10, 0.12, 0.14, 0.18, 0.24, 0.21])       # grey metamorphic host rock
MARBLE = np.array([0.30, 0.33, 0.35, 0.38, 0.46, 0.30])     # bright, strong CO3 dip in SWIR2
PEGMATITE = np.array([0.34, 0.36, 0.37, 0.39, 0.40, 0.38])  # bright, flat, iron-poor
MAFIC = np.array([0.05, 0.06, 0.06, 0.08, 0.13, 0.14])      # dark, ferrous


def _scene(n=90):
    rng = np.random.default_rng(1)
    refl = HOST[:, None, None] * (1 + 0.03 * rng.standard_normal((6, n, n)))
    refl[:, 10:25, 10:25] = MARBLE[:, None, None]
    refl[:, 40:55, 40:55] = PEGMATITE[:, None, None]
    refl[:, 40:55, 56:70] = MAFIC[:, None, None]
    yy, xx = np.mgrid[0:n, 0:n]
    dem = 3000 + xx * 12.0              # ~31 degree slope everywhere: exposed bedrock
    return refl.astype(np.float32), dem.astype(np.float32)


def test_models_light_up_the_right_rocks():
    refl, dem = _scene()
    usable = np.ones(refl.shape[1:], bool)
    raw = raw_evidence(refl)
    stats = EvidenceStats.fit({k: raw[k].ravel() for k in EVIDENCE})
    ev = evidence_layers(refl, usable, stats, 20.0, dem=dem)
    sc = model_scores(ev)
    k = [m.key for m in GEM_MODELS]
    marble, peg, contact, ultra = (sc[k.index(x)] for x in ("marble", "pegmatite", "contact", "ultramafic"))
    assert marble[17, 17] > 70 and marble[17, 17] > peg[17, 17]
    assert peg[47, 45] > 70 and peg[47, 45] > marble[47, 45]
    assert contact[47, 55] > contact[80, 80]         # pegmatite / mafic contact beats plain host rock
    assert marble[80, 80] < 30 and peg[80, 80] < 30


def test_flat_ground_is_not_bedrock():
    refl, dem = _scene()
    flat = np.full_like(dem, 2000)                   # a valley floor: bright fans, terraces, river beds
    raw = raw_evidence(refl)
    stats = EvidenceStats.fit({k: raw[k].ravel() for k in EVIDENCE})
    sc = model_scores(evidence_layers(refl, np.ones(dem.shape, bool), stats, 20.0, dem=flat))
    assert sc.max() == 0


def test_parse_occurrences_csv_and_geojson():
    occ = parse_occurrences(b"lat,lon,gem,name\n36.3,74.6,Ruby,Hunza A\n35.6,75.7,aquamarines,Shigar B\n35,74,moonstone,x\n")
    assert [o["model"] for o in occ] == ["marble", "pegmatite", None]
    gj = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [74.7, 35.9]}, "properties": {"gem": "emerald"}}]}
    occ = parse_occurrences(json.dumps(gj).encode(), "known.geojson")
    assert occ[0]["model"] == "contact" and occ[0]["lat"] == 35.9
    with pytest.raises(ValueError):
        parse_occurrences(b"lat,lon,gem\n95,74,ruby\n")


def test_success_rates():
    bg = np.arange(100, dtype=float)
    assert success_rates([95, 99], bg) == {"n": 2, "auc": 0.98, "top10": 1.0, "top20": 1.0}
    assert success_rates([5], bg)["auc"] < 0.1
    assert success_rates([], bg) == {"n": 0}


def test_region_gem_analysis_with_localities(small_scene, tmp_path):
    """Localities placed on the synthetic limestone must validate the marble model and train the ML model."""
    ref, rinfo = read_raster(small_scene["reference"])
    lab = np.nan_to_num(ref[0]).astype(np.uint8)
    w, s, e, n = rinfo.wgs84_bounds()
    aoi = {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}
    r = Region.create(tmp_path / "reg", RegionConfig(name="g", aoi=aoi, tile_size=64, source="local",
                                                     local_scenes=[str(small_scene["scene"])],
                                                     local_sensor="reflectance", local_dem=str(small_scene["dem"])))
    r.acquire(log=lambda m: None)
    from rasterio.warp import transform
    rng = np.random.default_rng(0)
    rr, cc = np.nonzero(lab == 1)
    pick = rng.choice(len(rr), 12, replace=False)
    xs, ys = rinfo.transform * (cc[pick] + 0.5, rr[pick] + 0.5)
    lons, lats = transform(rinfo.crs, "EPSG:4326", list(xs), list(ys))
    occ = [{"lat": la, "lon": lo, "gem": "ruby", "model": "marble", "name": f"L{i}"}
           for i, (lo, la) in enumerate(zip(lons, lats))]
    res = r.gem_analysis(occ, log=lambda m: None)
    marble = next(m for m in res["models"] if m["key"] == "marble")
    assert res["occurrences_inside"] >= 8
    assert marble["validation"]["n"] >= 8 and marble["validation"]["auc"] > 0.6
    assert res["ml"] and res["ml"]["occurrences"] >= 8
    assert (r.folder / "products" / "gem_targets.csv").exists()
    paths = r.build_mosaics()
    assert {"gems", "gem_marble", "gem_pegmatite", "gem_contact", "gem_ultramafic", "gem_ml"} <= set(paths)
    from rockmap.report import region_report
    assert region_report(r, tmp_path / "g.pdf", None, r.statistics(), gems=r.gems()).read_bytes()[:4] == b"%PDF"
