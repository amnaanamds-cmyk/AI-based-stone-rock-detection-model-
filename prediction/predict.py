"""Predict a lithology map with the trained model the application currently uses.

    python prediction/predict.py path/to/image.tif                       # -> results/<image name>/
    python prediction/predict.py image.tif --dem dem.tif --out my_results
    python prediction/predict.py image.tif --model model_v1 --algorithm rf

From Python (this is what the application does)::

    from prediction import Predictor
    p = Predictor()                     # loads models/trained/<current version>
    result = p.predict_file("image.tif", "results/run1", dem="dem.tif")

Flow: load model -> read image -> SAME preprocessing as in training (sensor scaling, masks, features,
normaliser - all read from the model's meta.json) -> predict block by block -> write the map.
This module never trains or changes a model.

Outputs in the output folder: classified.tif (class id per pixel, with colour table),
confidence.tif (0-100 %), classified.png / overlay previews, legend and area statistics (stats.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):                         # "python prediction/predict.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockmap import model_store                        # noqa: E402
from rockmap.config import class_name                  # noqa: E402
from rockmap.pipeline import ModelBundle, classify_scene  # noqa: E402
from rockmap.preprocessing import detect_sensor        # noqa: E402


class NoTrainedModel(RuntimeError):
    pass


class Predictor:
    """Loads one trained model version (default: the current one) for prediction only."""

    def __init__(self, version: Optional[str] = None, models_dir: Optional[Path] = None):
        root = Path(models_dir) if models_dir else model_store.models_dir()
        version = version or model_store.current_version(root)
        if not version or not (root / version / "meta.json").exists():
            raise NoTrainedModel(f"No trained model in {root}. Train one with: python training/train.py")
        self.version = version
        self.path = root / version
        self.bundle = ModelBundle(self.path)
        self.meta = self.bundle.meta

    @property
    def algorithm(self) -> str:
        return self.meta.get("app_algorithm") or self.bundle.best_algorithm()

    @property
    def classes(self) -> dict[int, str]:
        return {int(c): class_name(c) for c in self.meta["classes"]}

    @property
    def needs_dem(self) -> bool:
        return bool(self.meta["uses_dem"])

    @property
    def sample_data(self) -> bool:
        return bool(self.meta.get("sample_data"))

    def predict_file(self, image, out_dir, dem=None, algorithm: Optional[str] = None,
                     sensor: Optional[str] = None, progress=None) -> dict:
        """Classify an image file; returns the summary written by the pipeline (paths, statistics)."""
        if self.needs_dem and not dem:
            raise ValueError(f"{self.version} was trained with terrain features: give the DEM (dem=...)")
        sensor = sensor or (detect_sensor(image) if len(self.meta.get("sensors", [])) != 1
                            else self.meta.get("sensor"))
        kwargs = {"progress": progress} if progress else {}
        return classify_scene(self.path, image, out_dir, algorithm or self.algorithm,
                              dem_path=dem if self.needs_dem else None, sensor=sensor, **kwargs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict a lithology map with the trained model")
    ap.add_argument("image", help="multispectral GeoTIFF (6 bands: blue, green, red, NIR, SWIR1, SWIR2)")
    ap.add_argument("--dem", help="elevation model (needed when the model was trained with terrain features)")
    ap.add_argument("--out", help="output folder (default: results/<image name>)")
    ap.add_argument("--model", help="model version, e.g. model_v1 (default: the current model)")
    ap.add_argument("--algorithm", choices=["cnn", "rf", "svm"], help="default: the one chosen at training")
    ap.add_argument("--sensor", help="pixel scaling preset (default: same rule as in training)")
    a = ap.parse_args(argv)
    try:
        p = Predictor(a.model)
    except NoTrainedModel as e:
        sys.exit(str(e))
    out = Path(a.out) if a.out else Path("results") / Path(a.image).stem
    print(f"Model {p.version} ({p.algorithm}); classes: {', '.join(p.classes.values())}")
    if p.sample_data:
        print("WARNING: this model was trained on synthetic sample data - the map is a demonstration only.")
    result = p.predict_file(a.image, out, a.dem, a.algorithm, a.sensor)
    print(f"\nMean confidence {result['mean_confidence'] * 100:.0f} %, {result['masked_percent']:.1f} % masked "
          "(snow, water, vegetation, shadow, cloud)")
    for s in result.get("area_stats", []):
        if s["pixels"]:
            share = f"{s['percent']:5.1f} % of rock" if s.get("percent") is not None else "      (mask)"
            print(f"  {s['name']:34s} {s['area_km2']:10.2f} km2  {share}")
    print(f"\nMap written to {out / 'classified.tif'} (confidence: {out / 'confidence.tif'})")
    return result


if __name__ == "__main__":
    main()
