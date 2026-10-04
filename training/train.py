"""Train the lithology models on everything in the dataset folder and save a new model version.

    python training/train.py                         # uses the settings in training/config.py
    python training/train.py --data dataset/raw      # another dataset folder
    python training/train.py --epochs 50 --models cnn rf
    python training/train.py --check                 # only check the dataset, do not train

Steps (each is printed while it runs):

1. find the areas in the dataset folder and check them (bands, labels, overlap, class ids)
2. preprocess every area (reflectance, masks, features, block-split samples; cached)
3. train every model type in MODEL_TYPES on the pooled samples
4. evaluate on the held-out test blocks and write evaluation.txt
5. save everything as models/trained/model_vN (older versions are kept)
6. make model_vN the current model, so the application uses it from its next request

Every run trains from scratch on ALL areas in the folder (see training/README.md, "Retraining").
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):                         # "python training/train.py"
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training import config                            # noqa: E402  (also puts the project on sys.path)
from training.dataset import check_area, describe, find_areas  # noqa: E402
from training.evaluate import write_report             # noqa: E402
from training.preprocessing import (area_samples, describe_classes, pooled_collector,  # noqa: E402
                                    resolve_use_dem)

from rockmap import model_store                        # noqa: E402
from rockmap.config import ALGORITHMS                  # noqa: E402
from rockmap.pipeline import fit_bundle                # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Train RockMap lithology models (see training/README.md)")
    ap.add_argument("--data", default=str(config.DATA_DIR), help="dataset folder (default: %(default)s)")
    ap.add_argument("--models-dir", default=str(config.MODELS_DIR), help="where model versions are saved")
    ap.add_argument("--models", nargs="+", default=config.MODEL_TYPES, choices=ALGORITHMS,
                    help="model types to train (default: %(default)s)")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="CNN learning rate")
    ap.add_argument("--samples", type=int, default=config.SAMPLES_PER_CLASS, help="training pixels per class")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--no-activate", action="store_true", help="save the new version but keep the current model")
    ap.add_argument("--no-cache", action="store_true", help="recompute preprocessing for every area")
    ap.add_argument("--check", action="store_true", help="only check the dataset")
    return ap.parse_args(argv)


def main(argv=None) -> Path | None:
    a = parse_args(argv)
    data_dir, models_dir = Path(a.data), Path(a.models_dir)
    t0 = time.time()

    print(f"1/6  Dataset: {data_dir}")
    areas = find_areas(data_dir)
    if not areas:
        sys.exit(f"\nNo study area found in {data_dir}.\nPut each area in its own sub-folder with image.tif and "
                 "labels.tif (or labels.geojson) - see dataset/README.md.\nTo try the pipeline without real data: "
                 "python training/make_sample_dataset.py")
    print(describe(areas))
    bad = {a_.name: check_area(a_) for a_ in areas}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        msg = "\n".join(f"  {name}: " + "\n  ".join(" - " + p for p in problems) for name, problems in bad.items())
        sys.exit(f"\nThe dataset has problems - fix them and run again:\n{msg}")
    if a.check:
        print("\nDataset OK.")
        return None
    use_dem = resolve_use_dem(areas, config.USE_DEM)
    settings = {"samples_per_class": a.samples, "block_size": config.BLOCK_SIZE, "seed": a.seed,
                "split": [config.SPLIT["train"], config.SPLIT["val"], config.SPLIT["test"]],
                "patches": "cnn" in a.models, "use_dem": use_dem, "masking": config.MASK_SURFACES,
                "default_sensor": config.DEFAULT_SENSOR}

    print(f"\n2/6  Preprocessing {len(areas)} area(s)  (terrain features from DEM: {'yes' if use_dem else 'no'})")
    per_area = [area_samples(ar, settings, config.PROCESSED_DIR, config.USE_CACHE and not a.no_cache)
                for ar in areas]
    totals: dict[int, int] = {}
    for pa in per_area:
        for c, n in pa["class_pixels"].items():
            totals[int(c)] = totals.get(int(c), 0) + int(n)
    print(describe_classes(totals, config.CLASS_NAMES))
    if len(totals) < 2:
        sys.exit("\nAt least two classes with labelled pixels are needed to train a classifier.")
    collector = pooled_collector(per_area, settings)

    previous = model_store.current_version(models_dir)
    out_dir = model_store.next_version_dir(models_dir)
    print(f"\n3/6  Training {', '.join(a.models)} -> {out_dir.name}")
    meta = {"name": out_dir.name, "trained_with": "training module (training/train.py)",
            "sensor": per_area[0]["sensor"], "sensors": sorted({pa["sensor"] for pa in per_area}), "dos": False,
            "uses_dem": use_dem, "masking": config.MASK_SURFACES, "topo_correct": False,
            "sample_data": all(ar.is_sample for ar in areas),
            "dataset": [{"name": ar.name, "image": ar.image.name, "labels": ar.labels.name, "dem": bool(ar.dem),
                         "sample_data": ar.is_sample, "sensor": pa["sensor"], "labelled_pixels": pa["class_pixels"]}
                        for ar, pa in zip(areas, per_area)]}
    last = [0.0]

    def progress(f, msg):
        if time.time() - last[0] > 2 or f >= 1:      # keep the console readable
            print(f"     {msg}")
            last[0] = time.time()

    try:
        meta = fit_bundle(collector, out_dir, a.models, meta, a.epochs, a.seed, progress, 0.0, 1.0,
                          cnn_options={"batch_size": a.batch_size, "lr": a.lr, "width": config.CNN_WIDTH})
    except Exception:
        import shutil
        shutil.rmtree(out_dir, ignore_errors=True)    # never leave a half-written version behind
        raise

    print("\n4/6  Evaluation on held-out test blocks")
    if config.APP_MODEL_TYPE != "best" and config.APP_MODEL_TYPE in meta["algorithms"]:
        app_algo = config.APP_MODEL_TYPE
    else:
        app_algo = max(meta["algorithms"], key=lambda k: meta["algorithms"][k]["test_metrics"]["kappa"])
    meta["app_algorithm"] = app_algo
    meta["training_config"] = {"epochs": a.epochs, "batch_size": a.batch_size, "learning_rate": a.lr,
                               "cnn_width": config.CNN_WIDTH, "samples_per_class": a.samples,
                               "split": config.SPLIT, "block_size": config.BLOCK_SIZE, "seed": a.seed,
                               "model_types": a.models, "data_dir": str(data_dir),
                               "seconds": round(time.time() - t0, 1)}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "training_config.json").write_text(json.dumps(meta["training_config"], indent=2), encoding="utf-8")
    write_report(out_dir)
    for algo, m in meta["algorithms"].items():
        t = m["test_metrics"]
        flag = "  <- application" if algo == app_algo else ""
        print(f"     {algo:4s} accuracy {t['overall_accuracy'] * 100:5.1f} %   kappa {t['kappa']:.3f}   "
              f"macro F1 {t['macro_f1']:.3f}{flag}")
    print(f"     full report: {out_dir / 'evaluation.txt'}")

    print(f"\n5/6  Saved {out_dir}")
    if a.no_activate and previous:
        print(f"\n6/6  Not activated (--no-activate); the application keeps using {previous}.\n"
              f"     Activate later with: python training/models.py use {out_dir.name}")
    elif config.ACTIVATE_NEW_MODEL or not previous:
        model_store.set_current(out_dir.name, models_dir)
        print(f"\n6/6  {out_dir.name} is now the current model (previous: {previous or 'none'}). "
              f"The application picks it up automatically.")
    else:
        print(f"\n6/6  ACTIVATE_NEW_MODEL is False; current model stays {previous}.")
    if meta["sample_data"]:
        print("\nNOTE: trained on SYNTHETIC SAMPLE DATA only - this proves the pipeline works, it is NOT a "
              "real geological model. Put real data in dataset/raw and train again.")
    print(f"\nDone in {time.time() - t0:.0f} s.")
    return out_dir


if __name__ == "__main__":
    main()
