"""Command-line interface: ``rockmap <command>`` (or ``python -m rockmap <command>``)."""
from __future__ import annotations

import argparse
import json
import os
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
    app = create_app(Path(a.data) if a.data else None, workers=a.workers)
    url = f"http://{'127.0.0.1' if a.host in ('0.0.0.0', '::') else a.host}:{a.port}"
    print(f"RockMap dashboard on {url}  (data folder: {app.config['DATA_DIR']})", flush=True)
    if getattr(a, "open", False):
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    if a.debug:
        app.run(host=a.host, port=a.port, debug=True, threaded=True)
        return
    try:
        from waitress import serve
    except ImportError:  # pragma: no cover
        print("waitress not installed - using the Flask development server")
        app.run(host=a.host, port=a.port, threaded=True)
        return
    serve(app, host=a.host, port=a.port, threads=a.http_threads, channel_timeout=300)


# ---------------------------------------------------------------------------
# Region (large-area) commands
# ---------------------------------------------------------------------------

def _region(a):
    from .region import Region
    return Region(a.region)


def cmd_presets(a):
    from .presets import PRESETS
    for k, p in PRESETS.items():
        print(f"{k:18s} {p['name']}")


def cmd_region_create(a):
    from .presets import PRESETS
    from .region import Region, RegionConfig, load_geojson_geometry
    if a.preset:
        geom, name = PRESETS[a.preset]["geometry"], a.name or PRESETS[a.preset]["name"]
    elif a.aoi:
        geom, name = load_geojson_geometry(json.loads(Path(a.aoi).read_text())), a.name or Path(a.aoi).stem
    else:
        sys.exit("give --preset or --aoi")
    cfg = RegionConfig(name=name, aoi=geom, resolution=a.resolution, tile_size=a.tile_size,
                       years=a.years, months=a.months, max_cloud=a.max_cloud, max_scenes=a.max_scenes,
                       source="local" if a.local_scenes else "sentinel2", local_scenes=a.local_scenes or [],
                       local_sensor=a.local_sensor, local_dem=a.local_dem)
    r = Region.create(a.out, cfg)
    s = r.summary()
    print(f"Region '{name}': {s['tiles']} tiles of {cfg.tile_size * cfg.resolution / 1000:g} km, "
          f"EPSG:{r.config.epsg}, grid {r.width} x {r.height} px -> {a.out}")


def cmd_region_acquire(a):
    print(_region(a).acquire(_progress, keys=a.tiles, force=a.force))


def cmd_region_train(a):
    from .reference import load_reference_features
    r = _region(a)
    mapping = json.loads(Path(a.mapping).read_text()) if a.mapping else None
    feats = []
    for v in a.reference or []:
        feats += load_reference_features(v, a.field, mapping)
    meta = r.train(a.out, feats, a.label_raster or [], a.algorithms, a.samples, a.epochs, progress=_progress)
    _print_benchmark(meta)


def cmd_region_classify(a):
    print(_region(a).classify(a.model, a.algorithm, a.smoothing, _progress, keys=a.tiles))


def cmd_region_mosaic(a):
    for k, v in _region(a).build_mosaics(_progress).items():
        print(f"{k:12s} {v}")


def cmd_region_stats(a):
    r = _region(a)
    districts = json.loads(Path(a.districts).read_text()) if a.districts else None
    st = r.statistics(districts, a.name_field)
    print(f"{'Class':36s} {'km2':>10s} {'% rock':>7s}")
    for s in st["region"]:
        pct = f"{s['percent']:6.1f}" if s["percent"] is not None else "     -"
        print(f"{s['name']:36s} {s['area_km2']:10.2f} {pct}")
    for d in st["districts"]:
        print(f"\n{d['name']} ({d['km2']:.0f} km2)")
        for s in d["stats"]:
            if s["pixels"]:
                print(f"  {s['name']:34s} {s['area_km2']:10.2f}")
    if a.json:
        Path(a.json).write_text(json.dumps(st, indent=2))


def cmd_region_export(a):
    print(_region(a).export_geojson(a.out, a.min_pixels, a.include_masks, _progress))


def cmd_region_report(a):
    from .report import load_meta, region_report
    r = _region(a)
    districts = json.loads(Path(a.districts).read_text()) if a.districts else None
    model = a.model or r.state().get("model")
    print(region_report(r, a.out, load_meta(model) if model else None, r.statistics(districts, a.name_field),
                        a.organisation))


def cmd_region_query(a):
    print(json.dumps(_region(a).query(a.lon, a.lat), indent=2))


def cmd_region_status(a):
    r = _region(a)
    print(json.dumps({"name": r.config.name, **r.summary()}, indent=2))


def cmd_region_analyze(a):
    res = _region(a).analyze(a.units, _progress)
    print(f"\n{res['targets_total']} alteration targets, {len(res['clusters'])} spectral units")
    for t in res["targets_top"][:10]:
        print(f"  #{t['id']:<4d} {t['lat']:.5f}N {t['lon']:.5f}E  {t['type_label']:38s} score {t['mean_score']:5.1f}  "
              f"{t['area_ha']:6.1f} ha")
    from .analytics import HAZARD_CLASSES
    print("Landslide susceptibility (km2): " + ", ".join(
        f"{HAZARD_CLASSES[int(k)][0]} {v:,.1f}" for k, v in res["hazard_km2"].items()))


def cmd_region_label_units(a):
    mapping = dict(pair.split("=") for pair in a.assign)
    print(_region(a).label_clusters(mapping, _progress))


def cmd_quickstart(a):
    """Build a complete demo (synthetic region with every product) and start the dashboard."""
    from .quickstart import build_demo
    root = Path(a.data or os.environ.get("ROCKMAP_DATA_DIR", "data"))
    os.environ["ROCKMAP_DATA_DIR"] = str(root)
    build_demo(root, real=a.real, progress=_progress)
    if a.no_serve:
        return
    a.host, a.data, a.workers, a.http_threads, a.debug = a.host, str(root), None, 8, False
    cmd_serve(a)


def cmd_create_user(a):
    import getpass
    from .web.auth import create_user
    from .web.db import Database
    root = Path(a.data) if a.data else Path(os.environ.get("ROCKMAP_DATA_DIR", "data"))
    root.mkdir(parents=True, exist_ok=True)
    pw = a.password or getpass.getpass("Password: ")
    create_user(Database(root / "rockmap.db"), a.username, pw, a.role)
    print(f"user '{a.username}' ({a.role}) created")


def cmd_worker(a):
    from .web.app import create_app
    from .web.jobs import run_worker_forever
    app = create_app(Path(a.data) if a.data else None, workers=0)
    run_worker_forever(app, a.threads)


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
    s.add_argument("--debug", action="store_true", help="Flask development server with debugger")
    s.add_argument("--workers", type=int, default=None,
                   help="background job threads in this process (0 = use a separate 'rockmap worker')")
    s.add_argument("--http-threads", type=int, default=8)
    s.add_argument("--open", action="store_true", help="open the dashboard in the web browser")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("quickstart", help="build a complete demo (all features) and start the dashboard")
    s.add_argument("--data", help="data folder (default ./data)")
    s.add_argument("--real", action="store_true",
                   help="also download a real 40 x 40 km Gilgit region from Sentinel-2 (needs internet, ~5-10 min)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=5000)
    s.add_argument("--no-serve", action="store_true", help="only build the demo data")
    s.add_argument("--no-open", dest="open", action="store_false", help="do not open the browser")
    s.set_defaults(func=cmd_quickstart, open=True)

    s = sub.add_parser("worker", help="run background jobs (separate process from the web server)")
    s.add_argument("--data")
    s.add_argument("--threads", type=int, default=1)
    s.set_defaults(func=cmd_worker)

    s = sub.add_parser("create-user", help="create a dashboard user")
    s.add_argument("username")
    s.add_argument("--role", choices=["admin", "analyst", "viewer"], default="analyst")
    s.add_argument("--password", help="omit to be prompted")
    s.add_argument("--data")
    s.set_defaults(func=cmd_create_user)

    s = sub.add_parser("presets", help="list ready-made Gilgit-Baltistan study areas")
    s.set_defaults(func=cmd_presets)

    reg = sub.add_parser("region", help="large-area processing (e.g. all of Gilgit-Baltistan)")
    rsub = reg.add_subparsers(dest="region_command", required=True)

    s = rsub.add_parser("create", help="define a region and its tile grid")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--preset", help="see 'rockmap presets' (e.g. gilgit-baltistan, hunza, skardu)")
    g.add_argument("--aoi", help="GeoJSON boundary (EPSG:4326)")
    s.add_argument("--name")
    s.add_argument("--out", required=True, help="region folder")
    s.add_argument("--resolution", type=float, default=20.0)
    s.add_argument("--tile-size", type=int, default=1024)
    s.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    s.add_argument("--months", type=int, nargs="+", default=[7, 8, 9, 10], help="7-10 = least snow")
    s.add_argument("--max-cloud", type=float, default=30.0)
    s.add_argument("--max-scenes", type=int, default=6, help="scenes per tile composite")
    s.add_argument("--local-scenes", nargs="+", help="use these rasters instead of downloading Sentinel-2")
    s.add_argument("--local-sensor", default="sentinel2", choices=list(SENSORS))
    s.add_argument("--local-dem")
    s.set_defaults(func=cmd_region_create)

    def rarg(name, help_, func):
        s = rsub.add_parser(name, help=help_)
        s.add_argument("region", help="region folder")
        s.set_defaults(func=func)
        return s

    s = rarg("status", "tile counts", cmd_region_status)
    s = rarg("acquire", "download imagery + DEM and build cloud-free composites", cmd_region_acquire)
    s.add_argument("--tiles", nargs="+", help="only these tile keys (e.g. 004_002)")
    s.add_argument("--force", action="store_true", help="re-acquire tiles already done")
    s = rarg("train", "train models from reference geological maps", cmd_region_train)
    s.add_argument("--reference", nargs="+", help="GeoJSON / shapefile geological map(s) or training polygons")
    s.add_argument("--field", default="class_id")
    s.add_argument("--mapping", help="JSON: attribute value -> class id")
    s.add_argument("--label-raster", nargs="+", help="label GeoTIFF(s) with class ids")
    s.add_argument("--out", required=True)
    s.add_argument("--algorithms", nargs="+", default=list(ALGORITHMS), choices=ALGORITHMS)
    s.add_argument("--samples", type=int, default=4000)
    s.add_argument("--epochs", type=int, default=30)
    s = rarg("classify", "classify all acquired tiles", cmd_region_classify)
    s.add_argument("--model", required=True)
    s.add_argument("--algorithm", choices=ALGORITHMS)
    s.add_argument("--smoothing", type=int, default=3)
    s.add_argument("--tiles", nargs="+")
    s = rarg("analyze", "mineral-alteration targets, landslide susceptibility, spectral units", cmd_region_analyze)
    s.add_argument("--units", type=int, default=10, help="number of spectral units (2-16)")
    s = rarg("label-units", "turn spectral units into a lithology map", cmd_region_label_units)
    s.add_argument("assign", nargs="+", help="UNIT=CLASS pairs, e.g. 1=4 2=7 3=6")
    s = rarg("mosaic", "build region-wide GeoTIFF mosaics with overviews", cmd_region_mosaic)
    s = rarg("stats", "area statistics (optionally per district)", cmd_region_stats)
    s.add_argument("--districts", help="district boundaries GeoJSON")
    s.add_argument("--name-field", default="name")
    s.add_argument("--json")
    s = rarg("export", "export lithology polygons to GeoJSON", cmd_region_export)
    s.add_argument("--out", required=True)
    s.add_argument("--min-pixels", type=int, default=25)
    s.add_argument("--include-masks", action="store_true")
    s = rarg("report", "PDF report", cmd_region_report)
    s.add_argument("--out", required=True)
    s.add_argument("--model")
    s.add_argument("--districts")
    s.add_argument("--name-field", default="name")
    s.add_argument("--organisation", default="")
    s = rarg("query", "lithology, confidence and elevation at a point", cmd_region_query)
    s.add_argument("--lon", type=float, required=True)
    s.add_argument("--lat", type=float, required=True)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
