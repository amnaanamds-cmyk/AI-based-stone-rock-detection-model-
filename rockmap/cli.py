"""Command-line interface: ``rockmap <command>`` (or ``python -m rockmap <command>``)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ALGORITHM_LABELS, ALGORITHMS, SENSORS


def _progress(p: float, msg: str) -> None:
    print(f"[{p * 100:5.1f}%] {msg}", flush=True)


def cmd_synth(a):
    from .synthetic import write_scene
    paths = write_scene(a.out, a.size, a.size, a.seed, a.clouds)
    for k, v in paths.items():
        print(f"{k:10s} {v}")


def cmd_stack(a):
    from .preprocessing import find_sentinel2_bands, stack_bands
    extra = []
    if a.safe:
        bands, scl = find_sentinel2_bands(a.safe, int(a.resolution or 20))
        if scl is not None and a.scl:
            extra.append(scl)
    else:
        bands = a.bands
        if not bands or len(bands) != 6:
            sys.exit("give --safe or exactly 6 --bands (blue green red nir swir1 swir2)")
        if a.qa:
            extra.append(a.qa)
    out = stack_bands(bands, a.out, a.resolution, extra)
    print(f"stacked {len(bands)} bands{' + cloud layer' if extra else ''} -> {out}")


def cmd_align_dem(a):
    from .io import align_to, read_raster, write_raster
    _, info = read_raster(a.ref, bands=[1])
    dem = align_to(a.dem, info)
    print(write_raster(a.out, dem[None], info, descriptions=["elevation"]))


def cmd_rasterize(a):
    from .reference import rasterize_reference
    mapping = json.loads(Path(a.mapping).read_text()) if a.mapping else None
    print(rasterize_reference(a.vector, a.scene, a.out, a.field, mapping))


def cmd_train(a):
    from .evaluation import format_report
    from .pipeline import train_models
    meta = train_models(a.scene, a.labels, a.out, a.dem, a.cloud, a.algorithms, a.sensor, a.dos,
                        a.samples, a.epochs, a.block, a.seed, progress=_progress)
    print()
    for algo, m in meta["algorithms"].items():
        print(format_report(m["test_metrics"], f"{m['label']} (held-out spatial test blocks)"))
        print(f"Training time    : {m['train_seconds']:.1f} s\n")
    _print_benchmark(meta)


def _print_benchmark(meta):
    print("Benchmark summary")
    print(f"{'Algorithm':32s} {'OA %':>7s} {'Kappa':>7s} {'MacroF1':>8s} {'Train s':>8s}")
    for algo, m in meta["algorithms"].items():
        t = m["test_metrics"]
        print(f"{m['label']:32s} {t['overall_accuracy'] * 100:7.2f} {t['kappa']:7.3f} "
              f"{t['macro_f1']:8.3f} {m['train_seconds']:8.1f}")


def cmd_classify(a):
    from .evaluation import format_report
    from .pipeline import classify_scene
    res = classify_scene(a.model, a.scene, a.out, a.algorithm, a.dem, a.cloud, a.reference, a.window,
                         a.smoothing, progress=_progress)
    print(f"\nClassified {res['width']}x{res['height']} px with {res['algorithm_label']} in {res['seconds']} s")
    for s in res["area_stats"]:
        pct = f"{s['percent']:5.1f}%" if s["percent"] is not None else "   -  "
        print(f"  {s['name']:34s} {s['area_km2']:10.2f} km2  {pct}")
    if res["validation"]:
        print()
        print(format_report(res["validation"], "Validation against reference map"))
    print(f"\nOutputs written to {a.out}")


def cmd_evaluate(a):
    from .evaluation import compare_maps, format_report
    from .pipeline import load_labels
    from .io import read_raster
    pred, info = read_raster(a.pred, bands=[1])
    ref = load_labels(a.ref, info)
    m = compare_maps(pred[0].astype("uint8"), ref)
    print(format_report(m, "Map validation"))
    if a.json:
        Path(a.json).write_text(json.dumps(m, indent=2))


def cmd_demo(a):
    """Synthetic study area -> train all models -> classify -> validate."""
    from .synthetic import write_scene
    root = Path(a.out)
    print("1/3 Generating synthetic study area ...")
    paths = write_scene(root / "scene", a.size, a.size, a.seed, 0.03)
    print("2/3 Training CNN, Random Forest and SVM ...")
    from .pipeline import classify_scene, train_models
    meta = train_models(paths["scene"], paths["reference"], root / "model", paths["dem"],
                        epochs=a.epochs, samples_per_class=a.samples, progress=_progress, name="demo")
    _print_benchmark(meta)
    print("3/3 Classifying the full scene with every model ...")
    for algo in meta["algorithms"]:
        res = classify_scene(root / "model", paths["scene"], root / f"map_{algo}", algo, paths["dem"],
                             reference_path=paths["reference"])
        v = res["validation"]
        print(f"  {ALGORITHM_LABELS[algo]:32s} full-map agreement {v['overall_accuracy'] * 100:.2f}%  "
              f"kappa {v['kappa']:.3f}")
    print(f"\nDone. Outputs in {root}. Start the dashboard with:  rockmap serve")


def cmd_serve(a):
    from .web.app import create_app
    app = create_app(Path(a.data) if a.data else None)
    print(f"RockMap dashboard on http://{a.host}:{a.port}  (data folder: {app.config['DATA_DIR']})")
    app.run(host=a.host, port=a.port, debug=a.debug, threaded=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rockmap", description="AI-Based Satellite Rock/Stone Detection "
                                                            "and Geological Mapping System")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="generate a synthetic demo study area")
    s.add_argument("--out", default="data/demo")
    s.add_argument("--size", type=int, default=512)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--clouds", type=float, default=0.03, help="cloud cover fraction")
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("stack", help="stack band files (or a Sentinel-2 .SAFE folder) into one GeoTIFF")
    s.add_argument("--safe", help="unzipped Sentinel-2 L2A .SAFE folder")
    s.add_argument("--bands", nargs="+", help="6 band files in order blue green red nir swir1 swir2")
    s.add_argument("--qa", help="Landsat QA_PIXEL file to append as cloud layer")
    s.add_argument("--scl", action="store_true", help="append the Sentinel-2 SCL layer")
    s.add_argument("--resolution", type=float, help="output resolution in metres (default: first band)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_stack)

    s = sub.add_parser("align-dem", help="reproject a DEM (e.g. SRTM) onto a scene grid")
    s.add_argument("--dem", required=True)
    s.add_argument("--ref", required=True, help="scene GeoTIFF defining the target grid")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_align_dem)

    s = sub.add_parser("rasterize-map", help="burn a vector geological map into a label raster")
    s.add_argument("--vector", required=True, help="GeoJSON (or shapefile with fiona installed)")
    s.add_argument("--scene", required=True)
    s.add_argument("--field", required=True, help="attribute holding the unit / lithology")
    s.add_argument("--mapping", help="JSON file mapping attribute values to class ids 1-7")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_rasterize)

    def common(s, need_labels=False):
        s.add_argument("--scene", required=True, help="multi-band GeoTIFF (blue green red nir swir1 swir2)")
        s.add_argument("--dem", help="elevation GeoTIFF (any grid, it is re-projected)")
        s.add_argument("--cloud", help="cloud layer: Sentinel-2 SCL, Landsat QA_PIXEL or a binary mask")

    s = sub.add_parser("train", help="train the CNN and benchmark classifiers")
    common(s)
    s.add_argument("--labels", required=True, help="reference geological map raster (class ids 1-7)")
    s.add_argument("--out", required=True, help="output model folder")
    s.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    s.add_argument("--sensor", default="reflectance", choices=list(SENSORS))
    s.add_argument("--dos", action="store_true", help="apply dark object subtraction (TOA data)")
    s.add_argument("--samples", type=int, default=4000, help="training samples per class")
    s.add_argument("--epochs", type=int, default=30)
    s.add_argument("--block", type=int, default=32, help="spatial block size for train/test split")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("classify", help="produce a lithological map from a scene")
    common(s)
    s.add_argument("--model", required=True, help="trained model folder")
    s.add_argument("--algorithm", choices=ALGORITHMS, help="default: best model in the bundle")
    s.add_argument("--reference", help="reference map for validation")
    s.add_argument("--window", nargs=4, type=int, metavar=("COL", "ROW", "WIDTH", "HEIGHT"),
                   help="classify only this pixel window (region of interest)")
    s.add_argument("--smoothing", type=int, default=3, help="majority filter size (1 = off)")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_classify)

    s = sub.add_parser("evaluate", help="validate a classified map against a reference map")
    s.add_argument("--pred", required=True)
    s.add_argument("--ref", required=True)
    s.add_argument("--json", help="write metrics to this JSON file")
    s.set_defaults(func=cmd_evaluate)

    s = sub.add_parser("demo", help="run the full pipeline end-to-end on a synthetic study area")
    s.add_argument("--out", default=str(Path("data") / "demo_run"))
    s.add_argument("--size", type=int, default=512)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--epochs", type=int, default=25)
    s.add_argument("--samples", type=int, default=3000)
    s.set_defaults(func=cmd_demo)

    s = sub.add_parser("serve", help="start the web dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--data", help="data folder (default: $ROCKMAP_DATA_DIR or ./data)")
    s.add_argument("--debug", action="store_true")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
