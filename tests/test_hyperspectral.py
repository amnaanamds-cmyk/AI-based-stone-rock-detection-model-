import numpy as np
import pytest
import rasterio

from rockmap.hyperspectral import (MINERALS, classify, cube_wavelengths, map_scene, mineral_depths, needed_bands,
                                   parse_library, parse_wavelengths, sam)
from rockmap.io import read_raster

WL = np.arange(420, 2451, 10.0)
FEATURES = {
    "alunite": [(1762, .15, 12), (2165, .25, 15), (2320, .08, 15)],
    "kaolinite": [(2165, .15, 10), (2206, .25, 12)],
    "sericite": [(2200, .25, 14), (2350, .12, 15)],
    "chlorite": [(2250, .15, 12), (2335, .2, 15)],
    "calcite": [(2340, .3, 18)],
    "jarosite": [(2265, .2, 12)],
    "background": [],
}


def spectrum(name, noise=0.0, rng=None):
    r = 0.45 + 0.00003 * (WL - 400)
    for c, d, w in FEATURES[name]:
        r = r * (1 - d * np.exp(-((WL - c) / w) ** 2))
    if noise:
        r = r + rng.normal(0, noise, r.shape)
    return r


@pytest.mark.parametrize("noise", [0.0, 0.002])
def test_diagnostic_features_identify_minerals(noise):
    rng = np.random.default_rng(1)
    names = list(FEATURES)
    cube = np.stack([spectrum(n, noise, rng) for n in names], 1)[:, :, None]
    depths = mineral_depths({b: cube[b] for b in needed_bands(WL)}, WL)
    cls = classify(depths)[:, 0]
    keys = [m.key for m in MINERALS]
    got = [keys[c - 1] if c else "background" for c in cls]
    assert got == names


def test_sam_with_library_and_featureless_rock():
    lib = {k: spectrum(k) for k in ("kaolinite", "alunite", "calcite")}
    cube = np.stack([spectrum(n) for n in ("alunite", "calcite", "background")], 1)[:, :, None]
    cls, _ = sam(cube, WL, WL, lib)
    assert list(cls[:, 0]) == [2, 3, 0]


def test_wavelength_sources(tmp_path):
    assert parse_wavelengths("wavelength units = Micrometers\nwavelength = {0.42, 0.43, 0.44, 0.45, 0.46, "
                             "0.47, 0.48, 0.49, 0.50, 0.51}")[0] == pytest.approx(420)
    lib_wl, lib = parse_library(b"wavelength,kaolinite\n2.0,0.5\n2.2,0.4\n2.4,0.5\n")
    assert lib_wl[1] == pytest.approx(2200) and "kaolinite" in lib
    p = tmp_path / "c.tif"
    with rasterio.open(p, "w", driver="GTiff", width=2, height=2, count=len(WL), dtype="float32") as dst:
        for i, w in enumerate(WL, start=1):
            dst.set_band_description(i, f"{w:.0f} nm")
    assert np.allclose(cube_wavelengths(p), WL)


def _cube_like(info, path, pattern):
    """Write a (bands, H, W) reflectance x 10000 cube on ``info``'s grid; pattern maps rows to minerals."""
    h, w = info.height, info.width
    data = np.zeros((len(WL), h, w), np.uint16)
    for r in range(h):
        data[:, r, :] = (spectrum(pattern(r)) * 10000)[:, None]
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=len(WL), dtype="uint16",
                       crs=info.crs, transform=info.transform, nodata=0) as dst:
        dst.write(data)
        for i, wv in enumerate(WL, start=1):
            dst.set_band_description(i, f"{wv:.1f}")
    return path


def test_map_scene_and_region_integration(small_scene, tmp_path):
    _, info = read_raster(small_scene["scene"])
    cube = _cube_like(info, tmp_path / "enmap.tif", lambda r: "kaolinite" if r < info.height // 3 else
                      "sericite" if r < 2 * info.height // 3 else "background")
    s = map_scene(cube, tmp_path / "out.tif")
    assert s["pixels"]["kaolinite"] > 0 and s["pixels"]["sericite"] > 0 and s["pixels"]["calcite"] == 0

    from rockmap.region import Region, RegionConfig
    w, so, e, n = info.wgs84_bounds()
    aoi = {"type": "Polygon", "coordinates": [[[w, so], [e, so], [e, n], [w, n], [w, so]]]}
    r = Region.create(tmp_path / "reg", RegionConfig(name="h", aoi=aoi, tile_size=64, source="local",
                                                     local_scenes=[str(small_scene["scene"])],
                                                     local_sensor="reflectance", local_dem=str(small_scene["dem"])))
    r.acquire(log=lambda m: None)
    hinfo = r.import_hyperspectral([cube], log=lambda m: None)
    assert hinfo["mineral_km2"]["kaolinite"] > 0 and hinfo["mineral_km2"]["sericite"] > 0
    res = r.mineral_analysis(log=lambda m: None)
    assert res["hyperspectral"] is True
    assert {"hyper", "hyper_iron"} <= set(r.build_mosaics())
