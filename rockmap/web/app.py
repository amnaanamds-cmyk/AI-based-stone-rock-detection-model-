"""Flask web platform: authentication, scenes, regions, models, jobs, administration and REST API."""
from __future__ import annotations

import json
import logging
import os
import posixpath
import secrets
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request, send_from_directory,
                   session, url_for)
from rasterio.enums import Resampling
from werkzeug.utils import secure_filename

from .. import __version__
from ..config import (ALGORITHM_LABELS, ALGORITHMS, ALL_CLASSES, CLASS_BY_ID, MASK_CLASSES, ROCK_CLASSES,
                      SENSORS, data_dir)
from ..io import GeoInfo
from ..mapping import colorize, composite, hillshade_png, save_png
from ..preprocessing import to_reflectance
from . import auth
from .auth import audit, csrf_field, csrf_token, has_role, requires
from .db import Database, now
from .jobs import HANDLERS, WorkerPool, enqueue, handler, run_job, _claim

PREVIEW_SIZE = 1024
RASTER_EXT = {".tif", ".tiff", ".img", ".jp2", ".vrt"}
log = logging.getLogger("rockmap")


# ----------------------------------------------------------------------------- scene helpers
def register_scene(db: Database, root: Path, folder: Path, name: str, source: str, sensor: str, dos: bool,
                   user: Optional[str] = None) -> int:
    """Validate a scene folder, compute previews and store it in the database."""
    from ..io import align_to
    scene_path = folder / "scene.tif"
    with rasterio.open(scene_path) as src:
        if src.count < 6:
            raise ValueError(f"The scene has {src.count} bands; 6 are required "
                             "(blue, green, red, nir, swir1, swir2)")
        scale = max(1.0, max(src.width, src.height) / PREVIEW_SIZE)
        shape = (int(round(src.height / scale)), int(round(src.width / scale)))
        raw = src.read(list(range(1, 7)), out_shape=(6, *shape), resampling=Resampling.average).astype(np.float32)
        if src.nodata is not None:
            raw[raw == src.nodata] = np.nan
        info = GeoInfo(src.transform, src.crs, src.width, src.height)
    refl = to_reflectance(raw, sensor) if sensor != "reflectance" else raw
    valid = np.all(np.isfinite(refl), 0)
    save_png(composite(refl, (2, 1, 0), valid), folder / "rgb.png")
    save_png(composite(refl, (5, 3, 0), valid), folder / "falsecolor.png")
    if (folder / "dem.tif").exists():
        dem = align_to(folder / "dem.tif", info)
        save_png(hillshade_png(np.nan_to_num(dem, nan=float(np.nanmean(dem))), info.pixel_size[0] or 1.0),
                 folder / "hillshade.png", PREVIEW_SIZE)
    if (folder / "reference.tif").exists():
        ref = np.nan_to_num(align_to(folder / "reference.tif", info, Resampling.nearest)).astype(np.uint8)
        save_png(colorize(ref), folder / "reference.png", PREVIEW_SIZE)
    try:
        bounds = info.wgs84_bounds()
    except Exception:  # noqa: BLE001 - odd CRSs just disable the web map
        bounds = None
    return db.insert("scenes", name=name, source=source, sensor=sensor, dos=int(dos),
                     folder=folder.relative_to(root).as_posix(), has_dem=int((folder / "dem.tif").exists()),
                     has_reference=int((folder / "reference.tif").exists()),
                     has_cloud=int((folder / "cloud.tif").exists()), width=info.width, height=info.height,
                     crs=info.crs.to_string() if info.crs else None, bounds=bounds,
                     pixel_size=info.pixel_size[0], created_by=user)


def new_folder(root: Path, kind: str) -> Path:
    base = root / kind
    base.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name) for p in base.iterdir() if p.name.isdigit()] or [0])
    folder = base / str(n)
    folder.mkdir(parents=True)
    return folder


def _scene_file(root: Path, scene: dict, kind: str) -> Optional[Path]:
    p = root / scene["folder"] / f"{kind}.tif"
    return p if p.exists() else None


def model_summary(meta: dict) -> tuple[dict, str]:
    summary = {a: {"label": m["label"], "oa": m["test_metrics"]["overall_accuracy"],
                   "kappa": m["test_metrics"]["kappa"], "f1": m["test_metrics"]["macro_f1"],
                   "seconds": m["train_seconds"]} for a, m in meta["algorithms"].items()}
    return summary, max(summary, key=lambda a: summary[a]["kappa"])


# ----------------------------------------------------------------------------- scene job handlers
@handler("train")
def job_train(ctx):
    from ..pipeline import train_models
    db, root, p = ctx.db, ctx.root, ctx.params
    scene = db.get("scenes", ctx.job["scene_id"])
    folder = root / "models" / f"job{ctx.job['id']}"
    meta = train_models(_scene_file(root, scene, "scene"), _scene_file(root, scene, "reference"), folder,
                        dem_path=_scene_file(root, scene, "dem") if p["use_dem"] else None,
                        cloud_path=_scene_file(root, scene, "cloud"), algorithms=p["algorithms"],
                        sensor=scene["sensor"], dos=bool(scene["dos"]), samples_per_class=p["samples"],
                        epochs=p["epochs"], name=p["name"], progress=ctx.progress)
    summary, best = model_summary(meta)
    model_id = db.insert("models", name=p["name"], folder=folder.relative_to(root).as_posix(), scene_id=scene["id"],
                         uses_dem=int(meta["uses_dem"]), best_algo=best, summary=summary,
                         created_by=ctx.job["user"])
    db.update("jobs", ctx.job["id"], model_id=model_id, folder=folder.relative_to(root).as_posix())


@handler("classify")
def job_classify(ctx):
    from ..pipeline import classify_scene
    db, root, p = ctx.db, ctx.root, ctx.params
    scene = db.get("scenes", ctx.job["scene_id"])
    model = db.get("models", ctx.job["model_id"])
    folder = root / "jobs" / str(ctx.job["id"])
    db.update("jobs", ctx.job["id"], folder=folder.relative_to(root).as_posix())
    classify_scene(root / model["folder"], _scene_file(root, scene, "scene"), folder, p["algorithm"],
                   dem_path=_scene_file(root, scene, "dem") if model["uses_dem"] else None,
                   cloud_path=_scene_file(root, scene, "cloud"),
                   reference_path=_scene_file(root, scene, "reference") if p["validate"] else None,
                   window=p.get("window"), smoothing=p["smoothing"], progress=ctx.progress,
                   sensor=scene["sensor"])


# ----------------------------------------------------------------------------- app factory
def _secret_key(root: Path) -> str:
    if os.environ.get("ROCKMAP_SECRET_KEY"):
        return os.environ["ROCKMAP_SECRET_KEY"]
    p = root / "secret_key"
    if not p.exists():
        p.write_text(secrets.token_hex(32), encoding="utf-8")
        p.chmod(0o600)
    return p.read_text(encoding="utf-8").strip()


def _bootstrap_admin(db: Database, root: Path) -> None:
    if db.count("users"):
        return
    pw = os.environ.get("ROCKMAP_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
    # generated or demo passwords are temporary: the admin must replace them at first sign-in
    temporary = not os.environ.get("ROCKMAP_ADMIN_PASSWORD") or pw in auth.WEAK_PASSWORDS
    auth.create_user(db, os.environ.get("ROCKMAP_ADMIN_USER", "admin"), pw, "admin", must_change=temporary)
    if not os.environ.get("ROCKMAP_ADMIN_PASSWORD"):
        p = root / "initial_admin_password.txt"
        p.write_text(f"username: admin\npassword: {pw}\n(change it after the first login, then delete this file)\n", encoding="utf-8")
        p.chmod(0o600)
        print(f"\n*** First start: created user 'admin' with password '{pw}' (also saved in {p}) ***\n", flush=True)


def create_app(root: Optional[Path] = None, sync_jobs: bool = False, workers: Optional[int] = None) -> Flask:
    app = Flask(__name__)
    root = Path(root) if root else data_dir()
    for sub in ("scenes", "models", "jobs", "regions", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    app.config.update(
        DATA_DIR=root, SECRET_KEY=_secret_key(root), SYNC_JOBS=sync_jobs,
        MAX_CONTENT_LENGTH=int(os.environ.get("ROCKMAP_MAX_UPLOAD_MB", "4096")) * 1024 ** 2,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.environ.get("ROCKMAP_SECURE_COOKIES") == "1",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=12), CSRF_ENABLED=True,
        ORGANISATION=os.environ.get("ROCKMAP_ORGANISATION", ""),
    )
    db = Database(root / "rockmap.db")
    app.extensions["rockmap_db"] = db
    _bootstrap_admin(db, root)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("rasterio").setLevel(logging.WARNING)

    from .regions import bp as regions_bp
    from .field import bp as field_bp
    app.register_blueprint(regions_bp)
    app.register_blueprint(field_bp)

    if workers is None:
        workers = int(os.environ.get("ROCKMAP_WORKERS", "1"))
    if not sync_jobs and workers > 0:
        pool = WorkerPool(db, root, workers)
        pool.start()
        app.extensions["rockmap_workers"] = pool

    def submit(kind: str, params: dict, **cols) -> int:
        job_id = enqueue(db, kind, params, user=g.user["username"] if g.get("user") else None, **cols)
        audit(f"job.{kind}", f"job {job_id} {json.dumps(params)[:300]}")
        if app.config["SYNC_JOBS"]:
            job = _claim(db, "sync")
            while job:
                run_job(db, root, job)
                job = _claim(db, "sync")
        return job_id
    app.extensions["rockmap_submit"] = submit

    # ------------------------------------------------------------------ request hooks
    from .trained import is_training_module, sync_trained_model

    def sync_models(force=False):
        try:
            return sync_trained_model(db, force)
        except (OSError, ValueError, KeyError) as e:      # a broken model folder must not break the app
            logging.getLogger(__name__).warning("trained model not loaded: %s", e)
            return None
    sync_models(force=True)
    app.extensions["rockmap_sync_models"] = sync_models

    @app.before_request
    def before():
        sync_models()
        auth.load_user()
        auth.check_csrf()
        return auth.require_password_change()

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        return resp

    @app.context_processor
    def inject():
        return dict(version=__version__, rock_classes=ROCK_CLASSES, mask_classes=MASK_CLASSES,
                    algorithm_labels=ALGORITHM_LABELS, sensors=SENSORS, csrf_field=csrf_field,
                    csrf_token=csrf_token, current_user=g.get("user"), has_role=has_role,
                    organisation=app.config["ORGANISATION"])

    @app.template_filter("pct")
    def pct(v, digits=1):
        return "-" if v is None else f"{v * 100:.{digits}f}%"

    app.jinja_env.globals["class_by_id"] = ALL_CLASSES

    @app.errorhandler(403)
    def forbidden(_e):
        return render_template("error.html", code=403, message="You do not have permission for this action."), 403

    @app.errorhandler(404)
    def not_found(_e):
        if request.path.startswith("/api/"):
            return jsonify(error="not found"), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(400)
    def bad_request(e):
        if request.path.startswith("/api/"):
            return jsonify(error=str(e)), 400
        return render_template("error.html", code=400, message="Bad request - please go back and try again."), 400

    # ------------------------------------------------------------------ auth pages
    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            user, err = auth.authenticate(db, request.form.get("username", ""), request.form.get("password", ""))
            if user:
                auth.login_session(user)
                g.user = user
                audit("login")
                nxt = request.args.get("next") or ""
                return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("index"))
            flash(err, "error")
        return render_template("login.html")

    @app.post("/logout")
    def logout():
        if g.get("user"):
            audit("logout")
        session.clear()
        return redirect(url_for("login"))

    @app.route("/profile", methods=["GET", "POST"])
    @requires("viewer")
    def profile():
        token = None
        if request.method == "POST":
            action = request.form.get("action")
            if action == "password":
                from werkzeug.security import check_password_hash
                if not check_password_hash(g.user["password_hash"], request.form.get("current", "")):
                    flash("Current password is wrong.", "error")
                elif request.form.get("new") != request.form.get("confirm"):
                    flash("The new passwords do not match.", "error")
                else:
                    try:
                        auth.set_password(db, g.user["id"], request.form["new"])
                        auth.login_session(db.get("users", g.user["id"]))
                        audit("password.change")
                        flash("Password changed.", "ok")
                    except ValueError as e:
                        flash(str(e), "error")
            elif action == "token":
                token = auth.new_api_token(db, g.user["id"])
                audit("api_token.create")
            elif action == "revoke":
                db.update("users", g.user["id"], api_token_hash=None)
                audit("api_token.revoke")
                flash("API token revoked.", "ok")
        return render_template("profile.html", token=token, user=db.get("users", g.user["id"]))

    # ------------------------------------------------------------------ admin
    @app.route("/admin/users", methods=["GET", "POST"])
    @requires("admin")
    def admin_users():
        if request.method == "POST":
            action = request.form.get("action")
            try:
                if action == "create":
                    auth.create_user(db, request.form["username"], request.form["password"], request.form["role"],
                                     must_change=True)
                    audit("user.create", request.form["username"])
                    flash("User created.", "ok")
                else:
                    uid = int(request.form["user_id"])
                    target = db.get("users", uid) or abort(404)
                    if uid == g.user["id"] and action in ("role", "deactivate"):
                        raise ValueError("You cannot change your own role or deactivate yourself.")
                    if action == "role":
                        if request.form["role"] not in auth.ROLES:
                            abort(400)
                        db.update("users", uid, role=request.form["role"])
                    elif action == "deactivate":
                        db.update("users", uid, active=0, api_token_hash=None)
                    elif action == "activate":
                        db.update("users", uid, active=1)
                    elif action == "reset":
                        auth.set_password(db, uid, request.form["password"], must_change=True)
                    audit(f"user.{action}", target["username"])
                    flash("User updated.", "ok")
            except (ValueError, KeyError) as e:
                flash(str(e), "error")
            return redirect(url_for("admin_users"))
        return render_template("admin_users.html", users=db.all("users", order="username"), roles=auth.ROLES)

    @app.route("/admin/audit")
    @requires("admin")
    def admin_audit():
        return render_template("admin_audit.html", entries=db.all("audit", limit=500))

    # ------------------------------------------------------------------ dashboard
    @app.route("/")
    @requires("viewer")
    def index():
        from ..region import Region
        regions = []
        for r in db.all("regions"):
            try:
                regions.append({**r, "summary": Region(root / r["folder"]).summary()})
            except (OSError, ValueError, KeyError):
                continue
        return render_template("index.html", scenes=db.all("scenes"), models=db.all("models"),
                               jobs=db.all("jobs", limit=10), regions=regions,
                               running=db.count("jobs", "status IN ('queued','running')"))

    @app.post("/scenes/demo")
    @requires("analyst")
    def create_demo():
        from ..synthetic import write_scene
        seed = int(request.form.get("seed") or np.random.randint(0, 10_000))
        size = min(1024, max(128, int(request.form.get("size") or 512)))
        folder = new_folder(root, "scenes")
        write_scene(folder, size, size, seed, float(request.form.get("clouds") or 0.03))
        sid = register_scene(db, root, folder, f"Synthetic study area (seed {seed})", "demo", "reflectance", False,
                             g.user["username"])
        audit("scene.demo", str(sid))
        flash("Synthetic study area created. It includes a DEM and a reference geological map.", "ok")
        return redirect(url_for("scene_page", scene_id=sid))

    @app.route("/scenes/upload", methods=["GET", "POST"])
    @requires("analyst")
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
        folder = new_folder(root, "scenes")
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
            sid = register_scene(db, root, folder, name, "upload", sensor, bool(request.form.get("dos")),
                                 g.user["username"])
        except Exception as e:  # noqa: BLE001
            shutil.rmtree(folder, ignore_errors=True)
            flash(f"Upload failed: {e}", "error")
            return redirect(url_for("upload"))
        audit("scene.upload", f"{sid} {name}")
        flash("Scene uploaded.", "ok")
        return redirect(url_for("scene_page", scene_id=sid))

    @app.route("/scenes/<int:scene_id>")
    @requires("viewer")
    def scene_page(scene_id):
        scene = db.get("scenes", scene_id) or abort(404)
        return render_template("scene.html", scene=scene, models=db.all("models"),
                               jobs=db.all("jobs", "scene_id = ?", (scene_id,), limit=20))

    @app.post("/scenes/<int:scene_id>/delete")
    @requires("admin")
    def delete_scene(scene_id):
        scene = db.get("scenes", scene_id) or abort(404)
        shutil.rmtree(root / scene["folder"], ignore_errors=True)
        db.delete("scenes", scene_id)
        audit("scene.delete", scene["name"])
        flash("Scene deleted.", "ok")
        return redirect(url_for("index"))

    @app.post("/scenes/<int:scene_id>/classify")
    @requires("analyst")
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
        job_id = submit("classify", params, scene_id=scene_id, model_id=model["id"])
        return redirect(url_for("job_page", job_id=job_id))

    @app.post("/scenes/<int:scene_id>/train")
    @requires("analyst")
    def train(scene_id):
        scene = db.get("scenes", scene_id) or abort(404)
        if not scene["has_reference"]:
            flash("Training needs a reference geological map for this scene.", "error")
            return redirect(url_for("scene_page", scene_id=scene_id))
        algorithms = [a for a in request.form.getlist("algorithms") if a in ALGORITHMS] or list(ALGORITHMS)
        params = {"name": request.form.get("name") or f"Model from {scene['name']}", "algorithms": algorithms,
                  "epochs": min(200, max(1, int(request.form.get("epochs") or 25))),
                  "samples": min(50_000, max(100, int(request.form.get("samples") or 3000))),
                  "use_dem": bool(scene["has_dem"]) and bool(request.form.get("use_dem"))}
        job_id = submit("train", params, scene_id=scene_id)
        return redirect(url_for("job_page", job_id=job_id))

    # ------------------------------------------------------------------ jobs
    @app.route("/jobs")
    @requires("viewer")
    def jobs_page():
        return render_template("jobs.html", jobs=db.all("jobs", limit=200))

    @app.route("/jobs/<int:job_id>")
    @requires("viewer")
    def job_page(job_id):
        job = db.get("jobs", job_id) or abort(404)
        if job["region_id"]:
            return redirect(url_for("regions.region_page", region_id=job["region_id"], job=job_id))
        scene = db.get("scenes", job["scene_id"]) if job["scene_id"] else None
        result = meta = None
        if job["status"] == "done" and job["folder"]:
            folder = root / job["folder"]
            if job["kind"] == "classify" and (folder / "result.json").exists():
                result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            if job["kind"] == "train" and (folder / "meta.json").exists():
                meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        layers = []
        if result:
            base = url_for("files", relpath=job["folder"])
            layers.append({"name": "Lithology (classified)", "src": f"{base}/classified.png"})
            if result.get("validation"):
                layers.append({"name": "Agreement with reference", "src": f"{base}/agreement.png", "opacity": 0.6})
        return render_template("job.html", job=job, scene=scene, result=result, meta=meta, map_layers=layers,
                               model=db.get("models", job["model_id"]) if job["model_id"] else None,
                               log_tail=_log_tail(root, job_id))

    @app.post("/jobs/<int:job_id>/cancel")
    @requires("analyst")
    def cancel_job(job_id):
        job = db.get("jobs", job_id) or abort(404)
        if job["status"] == "queued":
            db.update("jobs", job_id, status="cancelled", finished=now(), message="Cancelled before start")
        elif job["status"] == "running":
            db.update("jobs", job_id, cancel=1, message="Cancelling...")
        audit("job.cancel", str(job_id))
        if request.path.startswith("/api/") or request.accept_mimetypes.best == "application/json":
            return jsonify(ok=True)
        return redirect(request.referrer or url_for("job_page", job_id=job_id))

    # ------------------------------------------------------------------ models
    @app.route("/models")
    @requires("viewer")
    def models_page():
        return render_template("models.html", models=db.all("models"))

    @app.route("/models/<int:model_id>")
    @requires("viewer")
    def model_page(model_id):
        model = db.get("models", model_id) or abort(404)
        meta = json.loads((root / model["folder"] / "meta.json").read_text(encoding="utf-8"))
        return render_template("model.html", model=model, meta=meta)

    @app.post("/models/<int:model_id>/delete")
    @requires("admin")
    def delete_model(model_id):
        model = db.get("models", model_id) or abort(404)
        if is_training_module(model):
            flash("Models from the training module are managed with: python training/models.py "
                  "(list / use / delete). Nothing was deleted.", "error")
            return redirect(url_for("models_page"))
        shutil.rmtree(root / model["folder"], ignore_errors=True)
        db.delete("models", model_id)
        audit("model.delete", model["name"])
        flash("Model deleted.", "ok")
        return redirect(url_for("models_page"))

    @app.route("/trained-models/<version>")
    @app.route("/trained-models/<version>/<name>")
    @requires("viewer")
    def trained_model_file(version, name=None):
        """Confusion matrices and reports of training-module models (models/trained/<version>/)."""
        from .. import model_store
        if not name or version not in model_store.versions() or not name.endswith((".png", ".txt", ".json")):
            abort(404)
        return send_from_directory(model_store.models_dir() / version, name)

    @app.route("/about")
    @requires("viewer")
    def about():
        return render_template("about.html")

    # ------------------------------------------------------------------ files & API
    @app.route("/files/<path:relpath>")
    @requires("viewer")
    def files(relpath):
        # only serve from the data sub-folders (no traversal out of them)
        norm = posixpath.normpath(relpath)
        if norm.startswith(("/", "..")) or norm.split("/", 1)[0] not in ("scenes", "jobs", "models", "regions", "observations"):
            abort(404)
        parts = norm.split("/")
        if parts[0] == "regions" and len(parts) > 2 and parts[2] in ("cache",):
            abort(404)
        return send_from_directory(root, norm, as_attachment=request.args.get("download") == "1")

    @app.route("/healthz")
    def healthz():
        try:
            db.count("users")
            ok = True
        except Exception:  # noqa: BLE001
            ok = False
        return jsonify(status="ok" if ok else "error", version=__version__,
                       queued=db.count("jobs", "status='queued'") if ok else None), (200 if ok else 503)

    @app.route("/api/jobs/<int:job_id>")
    @requires("viewer")
    def api_job(job_id):
        job = db.get("jobs", job_id) or abort(404)
        return jsonify({**{k: job[k] for k in ("id", "kind", "status", "progress", "message", "error",
                                               "region_id", "scene_id", "model_id", "created", "finished")},
                        "log": _log_tail(root, job_id, 15)})

    @app.post("/api/jobs/<int:job_id>/cancel")
    @requires("analyst")
    def api_cancel(job_id):
        return cancel_job(job_id)

    @app.route("/api/jobs")
    @requires("viewer")
    def api_jobs():
        return jsonify(db.all("jobs", limit=100))

    @app.route("/api/classes")
    @requires("viewer")
    def api_classes():
        return jsonify([{"id": c.id, "name": c.name, "color": c.color, "description": c.description,
                         "kind": "rock" if c.id in CLASS_BY_ID else "mask"} for c in ALL_CLASSES.values()])

    @app.route("/api/scenes")
    @requires("viewer")
    def api_scenes():
        return jsonify(db.all("scenes"))

    @app.route("/api/models")
    @requires("viewer")
    def api_models():
        return jsonify(db.all("models"))

    @app.route("/api/me")
    @requires("viewer")
    def api_me():
        return jsonify({k: g.user[k] for k in ("id", "username", "role")})

    return app


def _log_tail(root: Path, job_id: int, lines: int = 40) -> str:
    p = root / "logs" / f"job{job_id}.log"
    if not p.exists():
        return ""
    with open(p, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 20000))
        return "\n".join(fh.read().decode(errors="replace").splitlines()[-lines:])
