import json

import numpy as np
import rasterio

from rockmap.io import read_raster, write_raster
from rockmap.preprocessing import find_sentinel2_bands, stack_bands
from rockmap.reference import rasterize_reference


def test_rasterize_geojson(small_scene, tmp_path):
    # a polygon in UTM covering the upper-left quarter of the scene
    x0, y0 = 620000, 3810000
    poly = {"type": "Polygon", "coordinates": [[[x0, y0], [x0 + 1600, y0], [x0 + 1600, y0 - 1600],
                                                [x0, y0 - 1600], [x0, y0]]]}
    gj = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:32642"}},
          "features": [{"type": "Feature", "geometry": poly, "properties": {"unit": "Kohat Limestone"}}]}
    vec = tmp_path / "map.geojson"
    vec.write_text(json.dumps(gj))
    out = rasterize_reference(vec, small_scene["scene"], tmp_path / "ref.tif", "unit", {"Kohat Limestone": 1})
    lab, _ = read_raster(out)
    lab = np.nan_to_num(lab[0])
    assert (lab[:80, :80] == 1).all() and (lab[80:, 80:] == 0).all()


def test_stack_bands_and_find_safe(small_scene, tmp_path):
    data, info = read_raster(small_scene["scene"])
    safe = tmp_path / "S2B_MSIL2A.SAFE" / "GRANULE" / "IMG_DATA" / "R20m"
    for i, b in enumerate(["B02", "B03", "B04", "B08", "B11", "B12"]):
        write_raster(safe / f"T42SXD_20260301_{b}_20m.tif", (data[i] * 10000 + 1000).astype(np.uint16)[None], info)
    write_raster(safe / "T42SXD_20260301_SCL_20m.tif", np.full((1, 160, 160), 4, np.uint8), info)
    bands, scl = find_sentinel2_bands(tmp_path)
    assert [b.name.split("_")[2] for b in bands] == ["B02", "B03", "B04", "B08", "B11", "B12"]
    out = stack_bands(bands, tmp_path / "stack.tif", resolution=40, extra_paths=[scl])
    with rasterio.open(out) as src:
        assert src.count == 7 and src.width == 80 and abs(src.transform.a - 40) < 1e-6
