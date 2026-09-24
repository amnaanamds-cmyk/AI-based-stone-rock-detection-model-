"""Central configuration: lithology classes, sensor band presets and defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Lithology classes
# ---------------------------------------------------------------------------
# Class 0 is reserved for "no data / unlabelled". Broad lithology groups only,
# as defined in the project scope (no individual minerals).

NODATA_CLASS = 0
CLOUD_CLASS = 255  # written to output maps where pixels were masked (cloud / nodata)


@dataclass(frozen=True)
class RockClass:
    id: int
    name: str
    color: str  # hex colour used in maps and the dashboard legend
    description: str


ROCK_CLASSES: tuple[RockClass, ...] = (
    RockClass(1, "Limestone / Carbonate", "#4fa3e0",
              "Carbonate rocks (limestone, dolomite, marble). Bright, with a CO3 absorption near 2.33 um."),
    RockClass(2, "Sandstone", "#f2c14e",
              "Quartz-rich clastic sediment, often iron-stained (high red/blue ratio)."),
    RockClass(3, "Shale / Mudstone", "#8c6d46",
              "Fine clastic sediment with clay minerals (Al-OH absorption near 2.2 um)."),
    RockClass(4, "Granite / Felsic Igneous", "#e377c2",
              "Felsic intrusive rocks: granite, granodiorite. Moderately bright, flat spectrum."),
    RockClass(5, "Basalt / Mafic Igneous", "#3b3b3b",
              "Mafic / ultramafic rocks: basalt, gabbro, ophiolite. Dark, low albedo."),
    RockClass(6, "Metamorphic (Schist / Gneiss)", "#2ca02c",
              "Foliated metamorphic rocks: schist, gneiss, slate, phyllite."),
    RockClass(7, "Alluvium / Quaternary Deposits", "#d9d9a3",
              "Unconsolidated valley fill: gravel, sand, silt; often partly vegetated."),
)

CLASS_BY_ID = {c.id: c for c in ROCK_CLASSES}
CLASS_IDS = [c.id for c in ROCK_CLASSES]


def class_name(class_id: int) -> str:
    if class_id == CLOUD_CLASS:
        return "Masked (cloud / no data)"
    c = CLASS_BY_ID.get(int(class_id))
    return c.name if c else f"Class {class_id}"


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Sensors
# ---------------------------------------------------------------------------
# The system works on a canonical 6-band stack: blue, green, red, nir, swir1, swir2.
CANONICAL_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")


@dataclass(frozen=True)
class SensorPreset:
    name: str
    band_ids: tuple[str, ...]  # sensor band ids in canonical order
    scale: float               # reflectance = DN * scale + offset
    offset: float
    resolution: float          # native resolution (m) used when stacking


SENSORS: dict[str, SensorPreset] = {
    # Sentinel-2 L2A, processing baseline >= 04.00 (BOA_ADD_OFFSET = -1000)
    "sentinel2": SensorPreset("Sentinel-2 L2A", ("B02", "B03", "B04", "B08", "B11", "B12"),
                              scale=1e-4, offset=-0.1, resolution=20.0),
    # Sentinel-2 L2A, processing baseline < 04.00 (no offset)
    "sentinel2_legacy": SensorPreset("Sentinel-2 L2A (baseline < 04.00)",
                                     ("B02", "B03", "B04", "B08", "B11", "B12"),
                                     scale=1e-4, offset=0.0, resolution=20.0),
    # Landsat 8/9 Collection 2 Level-2 surface reflectance
    "landsat89": SensorPreset("Landsat 8/9 C2 L2", ("SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"),
                              scale=2.75e-5, offset=-0.2, resolution=30.0),
    # Data already expressed as reflectance in [0, 1] (e.g. the synthetic demo)
    "reflectance": SensorPreset("Surface reflectance (0-1)", CANONICAL_BANDS,
                                scale=1.0, offset=0.0, resolution=0.0),
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_PATCH_SIZE = 9          # CNN receptive field (pixels)
DEFAULT_SAMPLES_PER_CLASS = 4000
DEFAULT_BLOCK_SIZE = 32         # spatial block size (pixels) for train/test splitting
ALGORITHMS = ("cnn", "rf", "svm")
ALGORITHM_LABELS = {
    "cnn": "Convolutional Neural Network",
    "rf": "Random Forest",
    "svm": "Support Vector Machine",
}


def data_dir() -> Path:
    """Root folder for runtime data (uploaded scenes, jobs, models, database)."""
    root = Path(os.environ.get("ROCKMAP_DATA_DIR", Path.cwd() / "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root
