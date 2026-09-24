"""Flask web dashboard: manage study-area scenes, train models, run classification and view maps."""
from __future__ import annotations

import json
import posixpath
import shutil
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from flask import (Flask, abort, flash, jsonify, redirect, render_template, request,
                   send_from_directory, url_for)
from rasterio.enums import Resampling
from werkzeug.utils import secure_filename

from .. import __version__
from ..config import (ALGORITHM_LABELS, ALGORITHMS, CLASS_BY_ID, ROCK_CLASSES, SENSORS, data_dir)
from ..io import GeoInfo
from ..mapping import colorize, composite, hillshade_png, save_png
from ..preprocessing import to_reflectance
from .db import Database, now

PREVIEW_SIZE = 1024
RASTER_EXT = {".tif", ".tiff", ".img", ".jp2", ".vrt"}


def create_app(root: Optional[Path] = None, sync_jobs: bool = False) -> Flask:
    app = Flask(__name__)
    root = Path(root) if root else data_dir()
    for sub in ("scenes", "models", "jobs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    app.config.update(DATA_DIR=root, SECRET_KEY="rockmap-dashboard", SYNC_JOBS=sync_jobs,
                      MAX_CONTENT_LENGTH=2 * 1024 ** 3)
    db = Database(root / "rockmap.db")
    for j in db.all("jobs", "status IN ('queued', 'running')"):
        db.update("jobs", j["id"], status="failed", error="Interrupted by a server restart", finished=now())
    executor = ThreadPoolExecutor(max_workers=1)
    app.extensions["rockmap_db"] = db

    # ------------------------------------------------------------------ helpers
    def scene_dir(scene: dict) -> Path:
        return root / scene["folder"]

    def scene_file(scene: dict, kind: str) -> Optional[Path]:
        p = scene_dir(scene) / f"{kind}.tif"
        return p if p.exists() else None

    def submit(job_id: int, fn, *args) -> None:
        def run():
            db.update("jobs", job_id, status="running", message="Starting")

            def progress(p, msg):
                db.update("jobs", job_id, progress=float(p), message=msg)
            try:
                fn(job_id, progress, *args)
                db.update("jobs", job_id, status="done", progress=1.0, finished=now())
            except Exception as e:  # noqa: BLE001 - report any failure to the user
                traceback.print_exc()
                db.update("jobs", job_id, status="failed", error=f"{type(e).__name__}: {e}", finished=now())
        if app.config["SYNC_JOBS"]:
            run()
        else:
            executor.submit(run)

    def register_scene(folder: Path, name: str, source: str, sensor: str, dos: bool) -> int:
        """Validate the scene, compute previews and store it in the database."""
        scene_path = folder / "scene.tif"
        with rasterio.open(scene_path) as src:
            if src.count < 6:
                raise ValueError(f"The scene has {src.count} bands; 6 are required "
                                 "(blue, green, red, nir, swir1, swir2)")
            scale = max(1.0, max(src.width, src.height) / PREVIEW_SIZE)
            shape = (int(round(src.height / scale)), int(round(src.width / scale)))
            raw = src.read(list(range(1, 7)), out_shape=(6, *shape), resampling=Resampling.average)
            raw = raw.astype(np.float32)
            if src.nodata is not None:
                raw[raw == src.nodata] = np.nan
            info = GeoInfo(src.transform, src.crs, src.width, src.height)
        refl = to_reflectance(raw, sensor) if sensor != "reflectance" else raw
        valid = np.all(np.isfinite(refl), 0)
        save_png(composite(refl, (2, 1, 0), valid), folder / "rgb.png")
        save_png(composite(refl, (5, 3, 0), valid), folder / "falsecolor.png")
        if (folder / "dem.tif").exists():
            from ..io import align_to
            dem = align_to(folder / "dem.tif", info)
            save_png(hillshade_png(np.nan_to_num(dem, nan=float(np.nanmean(dem))), info.pixel_size[0] or 1.0),
                     folder / "hillshade.png", PREVIEW_SIZE)
        if (folder / "reference.tif").exists():
            from ..io import align_to
            ref = np.nan_to_num(align_to(folder / "reference.tif", info, Resampling.nearest)).astype(np.uint8)
            save_png(colorize(ref), folder / "reference.png", PREVIEW_SIZE)
        try:
            bounds = info.wgs84_bounds()
        except Exception:  # noqa: BLE001 - odd CRSs just disable the web map
            bounds = None
        return db.insert("scenes", name=name, source=source, sensor=sensor, dos=int(dos), folder=str(folder.relative_to(root)),
                         has_dem=int((folder / "dem.tif").exists()),
                         has_reference=int((folder / "reference.tif").exists()),
                         has_cloud=int((folder / "cloud.tif").exists()),
                         width=info.width, height=info.height,
                         crs=info.crs.to_string() if info.crs else None, bounds=bounds,
                         pixel_size=info.pixel_size[0])

    def new_folder(kind: str) -> Path:
        n = 1 + max([int(p.name) for p in (root / kind).iterdir() if p.name.isdigit()] or [0])
        folder = root / kind / str(n)
        folder.mkdir(parents=True)
        return folder

    # ------------------------------------------------------------------ job bodies
    def job_train(job_id, progress, scene, params):
        from ..pipeline import train_models
        folder = root / "models" / f"job{job_id}"
        meta = train_models(scene_file(scene, "scene"), scene_file(scene, "reference"), folder,
                            dem_path=scene_file(scene, "dem") if params["use_dem"] else None,
                            cloud_path=scene_file(scene, "cloud"), algorithms=params["algorithms"],
                            sensor=scene["sensor"], dos=bool(scene["dos"]),
                            samples_per_class=params["samples"], epochs=params["epochs"],
                            name=params["name"], progress=progress)
        summary = {a: {"label": m["label"], "oa": m["test_metrics"]["overall_accuracy"],
                       "kappa": m["test_metrics"]["kappa"], "f1": m["test_metrics"]["macro_f1"],
                       "seconds": m["train_seconds"]} for a, m in meta["algorithms"].items()}
        best = max(summary, key=lambda a: summary[a]["kappa"])
        model_id = db.insert("models", name=params["name"], folder=str(folder.relative_to(root)),
                             scene_id=scene["id"], uses_dem=int(meta["uses_dem"]), best_algo=best,
                             summary=summary)
        db.update("jobs", job_id, model_id=model_id, folder=str(folder.relative_to(root)))

    def job_classify(job_id, progress, scene, model, params):
        from ..pipeline import classify_scene
        folder = root / "jobs" / str(job_id)
        db.update("jobs", job_id, folder=str(folder.relative_to(root)))
        classify_scene(root / model["folder"], scene_file(scene, "scene"), folder, params["algorithm"],
                       dem_path=scene_file(scene, "dem") if model["uses_dem"] else None,
                       cloud_path=scene_file(scene, "cloud"),
                       reference_path=scene_file(scene, "reference") if params["validate"] else None,
                       window=params.get("window"), smoothing=params["smoothing"], progress=progress)

    # ------------------------------------------------------------------ template context
    @app.context_processor
    def inject():
        return dict(version=__version__, rock_classes=ROCK_CLASSES, algorithm_labels=ALGORITHM_LABELS,
                    sensors=SENSORS)

    @app.template_filter("pct")
    def pct(v, digits=1):
        return "-" if v is None else f"{v * 100:.{digits}f}%"

    # ------------------------------------------------------------------ pages
    @app.route("/")
    def index():
        return render_template("index.html", scenes=db.all("scenes"), models=db.all("models"),
                               jobs=db.all("jobs", limit=10))

    @app.post("/scenes/demo")
    def create_demo():
        from ..synthetic import write_scene
        seed = int(request.form.get("seed") or np.random.randint(0, 10_000))
        size = min(1024, max(128, int(request.form.get("size") or 512)))
        folder = new_folder("scenes")
        write_scene(folder, size, size, seed, float(request.form.get("clouds") or 0.03))
        sid = register_scene(folder, f"Synthetic study area (seed {seed})", "demo", "reflectance", False)
        flash("Synthetic study area created. It includes a DEM and a reference geological map.", "ok")
        return redirect(url_for("scene_page", scene_id=sid))

    @app.route("/scenes/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html")
        f = request.files.get("scene")
        if not f or not f.filename:
            flash("Please choose a multi-band scene GeoTIFF.", "error")
            return redirect(url_for("upload"))
        sensor = request.form.get("sensor", "reflectance")
        if sensor not in SENSORS:
            abort(400)
        folder = new_folder("scenes")
        try:
            for kind in ("scene", "dem", "cloud"):
                up = request.files.get(kind)
                if up and up.filename:
                    if Path(secure_filename(up.filename)).suffix.lower() not in RASTER_EXT:
                        raise ValueError(f"{kind}: unsupported file type (use GeoTIFF)")
                    up.save(folder / f"{kind}.tif")
            ref = request.files.get("reference")
            if ref and ref.filename:
                suffix = Path(secure_filename(ref.filename)).suffix.lower()
                if suffix in RASTER_EXT:
                    ref.save(folder / "reference.tif")
                elif suffix in (".geojson", ".json"):
                    from ..reference import rasterize_reference
                    vec = folder / "reference_map.geojson"
                    ref.save(vec)
                    mapping_txt = request.form.get("mapping", "").strip()
                    mapping = json.loads(mapping_txt) if mapping_txt else None
                    rasterize_reference(vec, folder / "scene.tif", folder / "reference.tif",
                                        request.form.get("field") or "class_id", mapping)
                else:
                    raise ValueError("reference map: use a GeoTIFF label raster or a GeoJSON file")
            name = request.form.get("name") or Path(f.filename).stem
            sid = register_scene(folder, name, "upload", sensor, bool(request.form.get("dos")))
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(folder, ignore_errors=True)
            flash(f"Upload failed: {e}", "error")
            return redirect(url_for("upload"))
        flash("Scene uploaded.", "ok")
        return redirect(url_for("scene_page", scene_id=sid))

    @app.route("/scenes/<int:scene_id>")
    def scene_page(scene_id):
        scene = db.get("scenes", scene_id) or abort(404)
        models = db.all("models")
        jobs = db.all("jobs", "scene_id = ?", (scene_id,), limit=20)
        return render_template("scene.html", scene=scene, models=models, jobs=jobs)

    @app.post("/scenes/<int:scene_id>/delete")
    def delete_scene(scene_id):
        scene = db.get("scenes", scene_id) or abort(404)
        shutil.rmtree(scene_dir(scene), ignore_errors=True)
        db.delete("scenes", scene_id)
        flash("Scene deleted.", "ok")
        return redirect(url_for("index"))

    @app.post("/scenes/<int:scene_id>/classify")
    def classify(scene_id):
        scene = db.get("scenes", scene_id) or abort(404)
        model = db.get("models", int(request.form.get("model_id") or 0))
        if not model:
            flash("Train or select a model first.", "error")
            return redirect(url_for("scene_page", scene_id=scene_id))
        if model["uses_dem"] and not scene["has_dem"]:
            flash("This model uses DEM terrain features but the scene has no DEM.", "error")
            return redirect(url_for("scene_page", scene_id=scene_id))
        algo = request.form.get("algorithm") or model["best_algo"]
        if algo not in (model["summary"] or {}):
            algo = model["best_algo"]
        window = None
        if request.form.get("window"):
            x, y, w, h = (int(float(v)) for v in request.form["window"].split(","))
            if w >= 16 and h >= 16:
                window = [x, y, w, h]
        params = {"algorithm": algo, "smoothing": int(request.form.get("smoothing") or 1),
                  "validate": bool(scene["has_reference"]), "window": window}
        job_id = db.insert("jobs", kind="classify", scene_id=scene_id, model_id=model["id"], params=params,
                           message="Queued")
        submit(job_id, job_classify, scene, model, params)
        return redirect(url_for("job_page", job_id=job_id))

    @app.post("/scenes/<int:scene_id>/train")
    def train(scene_id):
        scene = db.get("scenes", scene_id) or abort(404)
        if not scene["has_reference"]:
            flash("Training needs a reference geological map for this scene.", "error")
            return redirect(url_for("scene_page", scene_id=scene_id))
        algorithms = [a for a in request.form.getlist("algorithms") if a in ALGORITHMS] or list(ALGORITHMS)
        params = {"name": request.form.get("name") or f"Model from {scene['name']}",
                  "algorithms": algorithms,
                  "epochs": min(200, max(1, int(request.form.get("epochs") or 25))),
                  "samples": min(50_000, max(100, int(request.form.get("samples") or 3000))),
                  "use_dem": bool(scene["has_dem"]) and bool(request.form.get("use_dem"))}
        job_id = db.insert("jobs", kind="train", scene_id=scene_id, params=params, message="Queued")
        submit(job_id, job_train, scene, params)
        return redirect(url_for("job_page", job_id=job_id))

    @app.route("/jobs/<int:job_id>")
    def job_page(job_id):
        job = db.get("jobs", job_id) or abort(404)
        scene = db.get("scenes", job["scene_id"]) if job["scene_id"] else None
        result = meta = None
        if job["status"] == "done" and job["folder"]:
            folder = root / job["folder"]
            if job["kind"] == "classify" and (folder / "result.json").exists():
                result = json.loads((folder / "result.json").read_text())
            if job["kind"] == "train" and (folder / "meta.json").exists():
                meta = json.loads((folder / "meta.json").read_text())
        layers = []
        if result:
            base = url_for("files", relpath=job["folder"])
            layers.append({"name": "Lithology (classified)", "src": f"{base}/classified.png"})
            if result.get("validation"):
                layers.append({"name": "Agreement with reference", "src": f"{base}/agreement.png", "opacity": 0.6})
        return render_template("job.html", job=job, scene=scene, result=result, meta=meta, map_layers=layers,
                               model=db.get("models", job["model_id"]) if job["model_id"] else None)

    @app.route("/models")
    def models_page():
        return render_template("models.html", models=db.all("models"))

    @app.route("/models/<int:model_id>")
    def model_page(model_id):
        model = db.get("models", model_id) or abort(404)
        meta = json.loads((root / model["folder"] / "meta.json").read_text())
        return render_template("model.html", model=model, meta=meta)

    @app.post("/models/<int:model_id>/delete")
    def delete_model(model_id):
        model = db.get("models", model_id) or abort(404)
        shutil.rmtree(root / model["folder"], ignore_errors=True)
        db.delete("models", model_id)
        flash("Model deleted.", "ok")
        return redirect(url_for("models_page"))

    @app.route("/about")
    def about():
        return render_template("about.html")

    # ------------------------------------------------------------------ files & API
    @app.route("/files/<path:relpath>")
    def files(relpath):
        # only serve from the scenes / jobs / models folders (no traversal out of them)
        norm = posixpath.normpath(relpath)
        if norm.startswith(("/", "..")) or norm.split("/", 1)[0] not in ("scenes", "jobs", "models"):
            abort(404)
        relpath = norm
        return send_from_directory(root, relpath, as_attachment=request.args.get("download") == "1")

    @app.route("/api/jobs/<int:job_id>")
    def api_job(job_id):
        job = db.get("jobs", job_id) or abort(404)
        return jsonify({k: job[k] for k in ("id", "kind", "status", "progress", "message", "error")})

    @app.route("/api/classes")
    def api_classes():
        return jsonify([{"id": c.id, "name": c.name, "color": c.color, "description": c.description}
                        for c in ROCK_CLASSES])

    @app.route("/api/scenes")
    def api_scenes():
        return jsonify(db.all("scenes"))

    @app.route("/api/models")
    def api_models():
        return jsonify(db.all("models"))

    app.jinja_env.globals["class_by_id"] = CLASS_BY_ID
    return app
