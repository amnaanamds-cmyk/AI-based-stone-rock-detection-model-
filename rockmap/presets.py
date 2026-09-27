"""Ready-made study areas for Gilgit-Baltistan.

IMPORTANT: the Gilgit-Baltistan outline below is an *approximate* generalisation
(accuracy roughly +/- 5-15 km) drawn for processing extents only. It is not an official
boundary and must not be used for legal or administrative purposes. For official work,
upload the Survey of Pakistan / GB Government boundary (GeoJSON) when creating the region;
district boundaries can be uploaded as well to get per-district statistics.
"""
from __future__ import annotations

# (lon, lat), clockwise, starting at Shandur Pass
_GB_OUTLINE = [
    (72.55, 36.07), (72.52, 36.30), (72.72, 36.55), (73.00, 36.72), (73.25, 36.87), (73.48, 36.93),
    (73.80, 36.90), (74.10, 36.86), (74.45, 36.97), (74.62, 37.06), (74.87, 37.03), (75.10, 36.95),
    (75.43, 36.78), (75.75, 36.60), (76.00, 36.45), (76.18, 36.25), (76.40, 36.02), (76.53, 35.88),
    (76.72, 35.72), (76.92, 35.52), (76.95, 35.30), (76.82, 35.05), (76.55, 34.90), (76.25, 34.75),
    (75.90, 34.66), (75.55, 34.72), (75.20, 34.68), (74.95, 34.72), (74.60, 34.82), (74.30, 34.88),
    (74.00, 34.95), (73.77, 35.14), (73.55, 35.25), (73.30, 35.30), (73.02, 35.42), (72.85, 35.62),
    (72.72, 35.86), (72.55, 36.07),
]

GILGIT_BALTISTAN = {
    "type": "Polygon",
    "coordinates": [[list(p) for p in _GB_OUTLINE]],
}


def _box(lon: float, lat: float, half_km: float) -> dict:
    dlat = half_km / 111.32
    import math
    dlon = half_km / (111.32 * math.cos(math.radians(lat)))
    ring = [[lon - dlon, lat - dlat], [lon + dlon, lat - dlat], [lon + dlon, lat + dlat],
            [lon - dlon, lat + dlat], [lon - dlon, lat - dlat]]
    return {"type": "Polygon", "coordinates": [ring]}


PRESETS: dict[str, dict] = {
    "gilgit-baltistan": {
        "name": "Gilgit-Baltistan (whole region, approximate outline)",
        "geometry": GILGIT_BALTISTAN,
        "note": "~73,000 km2, about 200 tiles of 20 km at 20 m. Full acquisition downloads several GB "
                "of imagery and takes hours; start with a sub-area to calibrate the model.",
    },
    "gilgit": {"name": "Gilgit city & surroundings (40 x 40 km)", "geometry": _box(74.31, 35.92, 20)},
    "hunza": {"name": "Hunza valley - Karimabad / Aliabad (40 x 40 km)", "geometry": _box(74.66, 36.32, 20)},
    "skardu": {"name": "Skardu basin (40 x 40 km)", "geometry": _box(75.63, 35.30, 20)},
    "astore": {"name": "Astore valley / Nanga Parbat east (40 x 40 km)", "geometry": _box(74.86, 35.37, 20)},
    "chilas": {"name": "Chilas - Indus valley (40 x 40 km)", "geometry": _box(74.10, 35.42, 20)},
    "khaplu": {"name": "Khaplu - Shyok valley (40 x 40 km)", "geometry": _box(76.33, 35.16, 20)},
    "shigar": {"name": "Shigar valley - pegmatite gem belt (40 x 40 km)", "geometry": _box(75.70, 35.55, 20),
               "note": "Aquamarine, topaz, tourmaline and garnet pegmatites of the Shigar / Braldu valleys."},
    "hunza-gems": {"name": "Hunza ruby & spinel marble belt (40 x 40 km)", "geometry": _box(74.66, 36.32, 20),
                   "note": "Marble-hosted ruby and spinel of central Hunza (Karimabad - Aliabad - Ganesh)."},
    "gilgit-small": {"name": "Gilgit city (10 x 10 km quick test)", "geometry": _box(74.31, 35.92, 5)},
}
