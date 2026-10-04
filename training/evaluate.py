"""Evaluate a trained model and write a readable report.

Two kinds of evaluation:

* **held-out test blocks** (automatic after every training run): metrics on pixels from spatial
  blocks the models never saw during training. Written to ``evaluation.txt`` / ``evaluation.json``
  in the model folder.
* **independent area** (optional): classify a whole labelled area that was NOT used for training and
  compare it with its labels - the most honest check::

      python training/evaluate.py --data path/to/area_folder            # current model
      python training/evaluate.py --model model_v2 --data path/to/area_folder

Metrics (pixel classification): overall accuracy, Cohen's kappa (agreement beyond chance; > 0.6 good,
> 0.8 very good), per-class precision / recall / F1, macro F1 and the confusion matrix.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):                         # "python training/evaluate.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockmap import model_store                        # noqa: E402
from rockmap.config import ALGORITHM_LABELS, CLASS_BY_ID  # noqa: E402
from rockmap.evaluation import compare_maps, format_report  # noqa: E402

GOOD_KAPPA = 0.6


def confusion_text(m: dict) -> str:
    classes = [c["id"] for c in m["per_class"]]
    short = [CLASS_BY_ID[c].name.split(" ")[0][:9] if c in CLASS_BY_ID else str(c) for c in classes]
    lines = ["Confusion matrix (rows = reference, columns = predicted):",
             " " * 11 + "".join(f"{s:>10s}" for s in short)]
    for s, row in zip(short, m["confusion_matrix"]):
        lines.append(f"{s:>10s} " + "".join(f"{v:>10d}" for v in row))
    return "\n".join(lines)


def warnings_for(meta: dict) -> list[str]:
    out = []
    if meta.get("sample_data"):
        out.append("Trained on SYNTHETIC SAMPLE DATA: the numbers below only prove that the pipeline works. "
                   "They say nothing about real geology. Train on a real dataset before using the maps.")
    algo = meta.get("app_algorithm")
    m = meta["algorithms"].get(algo, {}).get("test_metrics") if algo else None
    if m:
        if m["kappa"] < GOOD_KAPPA:
            out.append(f"Kappa {m['kappa']:.2f} is below {GOOD_KAPPA}: the map is unreliable. Add more / better "
                       "labelled data, check the labels, or merge classes that are hard to separate.")
        weak = [c["name"] for c in m["per_class"] if c["support"] and c["f1"] < 0.5]
        if weak:
            out.append("Classes with F1 below 0.5 (often confused): " + ", ".join(weak))
        missing = [CLASS_BY_ID[c].name for c in meta["classes"] if not any(p["id"] == c and p["support"]
                                                                          for p in m["per_class"])]
        if missing:
            out.append("Classes without test pixels (not evaluated): " + ", ".join(missing))
    if meta.get("n_train", 0) < 2000:
        out.append(f"Only {meta.get('n_train', 0)} training pixels: results will vary a lot between runs.")
    return out


def write_report(model_dir: Path) -> str:
    """Write evaluation.txt / evaluation.json for a model folder and return the text."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    parts = [f"Model {model_dir.name} - trained {meta['created']}",
             f"Training data: {', '.join(a['name'] for a in meta.get('dataset', [])) or meta.get('scene', '?')}",
             f"Samples: {meta['n_train']:,} train / {meta['n_val']:,} validation / {meta['n_test']:,} test pixels "
             f"(spatial blocks of {meta['block_size']} px)", ""]
    for w in warnings_for(meta):
        parts.append("WARNING: " + w)
    if parts[-1] != "":
        parts.append("")
    summary = {}
    for algo, a in meta["algorithms"].items():
        m = a["test_metrics"]
        summary[algo] = {"overall_accuracy": m["overall_accuracy"], "kappa": m["kappa"], "macro_f1": m["macro_f1"]}
        used = "   <- used by the application" if algo == meta.get("app_algorithm") else ""
        parts += [format_report(m, f"{ALGORITHM_LABELS.get(algo, algo)} ({algo}){used}"), "", confusion_text(m), ""]
    text = "\n".join(parts)
    (model_dir / "evaluation.txt").write_text(text, encoding="utf-8")
    (model_dir / "evaluation.json").write_text(json.dumps({"model": model_dir.name, "app_algorithm":
                                                           meta.get("app_algorithm"), "summary": summary,
                                                           "warnings": warnings_for(meta)}, indent=2),
                                               encoding="utf-8")
    return text


def evaluate_on_area(model_dir: Path, area, log=print) -> dict:
    """Classify an independent labelled area with the model and compare with its labels."""
    import numpy as np
    from rockmap.io import read_raster
    from rockmap.pipeline import ModelBundle, classify_scene

    from training.preprocessing import label_array, sensor_for

    bundle = ModelBundle(model_dir)
    algo = bundle.meta.get("app_algorithm") or bundle.best_algorithm()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        classify_scene(model_dir, area.image, tmp, algo, dem_path=area.dem if bundle.meta["uses_dem"] else None,
                       sensor=sensor_for(area, bundle.meta.get("sensor", "auto")))
        pred, info = read_raster(tmp / "classified.tif")
        ref = label_array(area, info, tmp)
    m = compare_maps(pred[0].astype(np.uint8), ref)
    text = format_report(m, f"{model_dir.name} ({algo}) on independent area '{area.name}'") + "\n\n" + confusion_text(m)
    out = model_dir / f"evaluation_{area.name}.txt"
    out.write_text(text, encoding="utf-8")
    log(text)
    log(f"\nsaved {out}")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate a trained RockMap model")
    ap.add_argument("--model", default="current", help="model version (e.g. model_v2) or 'current'")
    ap.add_argument("--data", help="an independent labelled area folder (image.tif + labels) to test on")
    ap.add_argument("--models-dir", help="models folder (default: models/trained)")
    a = ap.parse_args(argv)
    root = Path(a.models_dir) if a.models_dir else model_store.models_dir()
    version = model_store.current_version(root) if a.model == "current" else a.model
    if not version or not (root / version / "meta.json").exists():
        sys.exit(f"No trained model '{a.model}' in {root}. Train one first: python training/train.py")
    if a.data:
        from training.dataset import check_area, find_areas
        areas = find_areas(Path(a.data))
        if not areas:
            sys.exit(f"No area (folder with image.tif) found in {a.data}")
        for area in areas:
            problems = check_area(area)
            if problems:
                sys.exit(f"{area.name}: " + "; ".join(problems))
            evaluate_on_area(root / version, area)
    else:
        print(write_report(root / version))


if __name__ == "__main__":
    main()
