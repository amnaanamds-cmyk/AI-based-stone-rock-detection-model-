import os

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from rockmap.acquisition import (S2Item, Season, build_composite, copernicus_dem_urls, mgrs_id,
                                 mgrs_tiles_for_bounds, utm_epsg)
from rockmap.config import SNOW_CLASS, VEGETATION_CLASS, WATER_CLASS
from rockmap.io import GeoInfo
from rockmap.preprocessing import c_correction, illumination, landcover_mask


def test_utm_and_mgrs_ids():
    assert utm_epsg(74.3, 35.9) == 32643          # Gilgit
    assert utm_epsg(71.7, 34.1) == 32642          # Charsadda
    # Sentinel-2 granule 43SDA has its upper-left corner at (399960, 4100040)
    assert mgrs_id(43, 450_000, 4_050_000, 36.5) == "43SDA"
    assert "43SDV" in mgrs_tiles_for_bounds(74.2, 35.85, 74.4, 36.0)        # Gilgit city
    gb = mgrs_tiles_for_bounds(72.5, 34.6, 77.0, 37.1)
    assert all(t.startswith(("42", "43", "44")) for t in gb) and len(gb) > 15


def test_dem_urls():
    urls = copernicus_dem_urls((74.2, 35.5, 75.3, 36.2))
    assert len(urls) == 4
    assert urls[0].endswith("Copernicus_DSM_COG_10_N35_00_E074_00_DEM/Copernicus_DSM_COG_10_N35_00_E074_00_DEM.tif")


def _stac(offset_applied):
    return {"id": "S2B_43SDV_20240921_0_L2A", "bbox": [74, 35, 75, 36],
            "properties": {"datetime": "2024-09-21T05:59:00Z", "eo:cloud_cover": 3.2, "view:sun_azimuth": 150,
                           "view:sun_elevation": 50, "proj:epsg": 32643, "earthsearch:boa_offset_applied": offset_applied,
                           "mgrs:utm_zone": 43, "mgrs:latitude_band": "S", "mgrs:grid_square": "DV"},
            "assets": {k: {"href": f"https://x/{k}.tif", "raster:bands": [{"scale": 0.0001, "offset": -0.1}]}
                       for k in ("blue", "green", "red", "nir", "swir16", "swir22", "scl")}}


def test_s2item_offset_handling():
    # Earth Search harmonised items already had BOA_ADD_OFFSET removed from the pixel values
    assert S2Item.from_stac(_stac(True)).offset == 0.0
    it = S2Item.from_stac(_stac(False))
    assert it.offset == -0.1 and it.mgrs == "43SDV" and it.date == "2024-09-21" and len(it.hrefs) == 7


def test_season_ranges():
    r = Season(years=(2024,), months=(2, 9)).ranges()
    assert r == [("2024-02-01", "2024-02-29"), ("2024-09-01", "2024-09-30")]


def _grid(n=40):
    return GeoInfo(from_origin(400000, 4000000, 20, 20), CRS.from_epsg(32643), n, n)


def test_build_composite_median_cloud_and_snow():
    grid = _grid()
    rock = np.array([0.10, 0.14, 0.17, 0.21, 0.32, 0.33], np.float32)
    snow = np.array([0.80, 0.78, 0.75, 0.70, 0.08, 0.06], np.float32)

    def make(scl_value_left, scl_value_right, spectrum_right, noise):
        refl = np.broadcast_to(rock[:, None, None], (6, 40, 40)).copy() + noise
        refl[:, :, 20:] = spectrum_right[:, None, None]
        scl = np.full((40, 40), scl_value_left, np.uint8)
        scl[:, 20:] = scl_value_right
        return refl, scl

    scenes = {
        "a": make(5, 11, snow, 0.00),      # left clear, right snow
        "b": make(9, 11, snow, 0.10),      # left cloud (must be ignored), right snow
        "c": make(5, 11, snow, 0.02),      # left clear
    }
    items = [S2Item(k, f"2024-08-0{i + 1}T00:00:00Z", 5.0, 150, 55, [], 32643, {}) for i, k in enumerate(scenes)]
    stack, rep = build_composite(grid, items, None, max_items=3, target_clear=2.0, log=lambda m: None,
                                 reader=lambda item, g: scenes[item.id])
    assert rep.items == ["a", "b", "c"]
    left = stack[:6, 5, 5] / 10000
    assert np.allclose(left, rock + 0.01, atol=2e-4)       # median of clear obs (0.00, 0.02), cloud ignored
    assert (stack[6, :, :20] == 4).all() and (stack[6, :, 20:] == 11).all()
    assert rep.snow_fraction == pytest.approx(0.5)


def test_landcover_rules():
    r = np.zeros((6, 1, 5), np.float32)
    r[:, 0, 0] = [0.10, 0.14, 0.17, 0.21, 0.32, 0.33]   # rock
    r[:, 0, 1] = [0.80, 0.78, 0.75, 0.70, 0.08, 0.06]   # snow
    r[:, 0, 2] = [0.06, 0.07, 0.05, 0.03, 0.02, 0.01]   # water
    r[:, 0, 3] = [0.03, 0.07, 0.04, 0.40, 0.20, 0.10]   # vegetation
    r[:, 0, 4] = [0.01, 0.01, 0.01, 0.02, 0.03, 0.03]   # deep shadow
    lc = landcover_mask(r)[0]
    assert lc.tolist() == [0, SNOW_CLASS, WATER_CLASS, VEGETATION_CLASS, 253]


def test_c_correction_flattens_topography():
    yy, xx = np.mgrid[0:120, 0:120].astype(np.float32)
    dem = 1000 + 400 * np.sin(xx / 12) * np.cos(yy / 15)
    cos_i = illumination(dem, 20.0, 150.0, 50.0)
    true = np.full((6, 120, 120), 0.2, np.float32)
    shaded = true * (0.2 + 0.8 * np.clip(cos_i, 0, 1))[None]
    corr = c_correction(shaded, cos_i, 50.0)
    assert np.std(corr[0]) < 0.5 * np.std(shaded[0])


@pytest.mark.skipif(not os.environ.get("ROCKMAP_NETWORK_TESTS"), reason="set ROCKMAP_NETWORK_TESTS=1 to run")
def test_real_acquisition_gilgit(tmp_path):
    from rasterio.warp import transform
    from rockmap.acquisition import S2Catalog, fetch_dem
    x, y = transform("EPSG:4326", "EPSG:32643", [74.30], [35.95])
    grid = GeoInfo(from_origin(round(x[0] / 20) * 20, round(y[0] / 20) * 20, 20, 20), CRS.from_epsg(32643), 128, 128)
    dem = fetch_dem(grid)
    assert 1200 < np.nanmin(dem) and np.nanmax(dem) < 6000
    items = S2Catalog(cache_dir=tmp_path).search(grid.wgs84_bounds(), Season((2024,), (9,), 20))
    assert items
    stack, rep = build_composite(grid, items, dem, max_items=1)
    assert rep.clear_fraction > 0.5 and 300 < np.median(stack[2][stack[2] > 0]) < 4000
