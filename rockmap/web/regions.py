"""Region pages, map tiles, training-area annotations and region job handlers."""
from __future__ import annotations

import csv
import io
import json
import os
import shutil
from pathlib import Path

from flask import (Blueprint, Response, abort, current_app, flash, g, jsonify, redirect, render_template,
                   request, url_for)
from werkzeug.utils import secure_filename

from ..config import ALGORITHM_LABELS, ALGORITHMS, ALL_CLASSES, CLASS_IDS, CLOUD_CLASS, ROCK_CLASSES
from ..presets import PRESETS
from ..region import Region, RegionConfig, load_geojson_geometry
from .auth import audit, has_role, requires
from .db import Database
from .jobs import handler

bp = Blueprint("regions", __name__)
LAYERS = ("rgb", "falsecolor", "hillshade", "surface", "lithology", "confidence")


def _db() -> Database:
    return current_app.extensions["rockmap_db"]


def _root() -> Path:
    return current_app.config["DATA_DIR"]


def _region_row(region_id: int) -> dict:
    return _db().get("regions", region_id) or abort(404)


def _region(row: dict, root: Path | None = None) -> Region:
    return Region((root or _root()) / row["folder"])


def _references(region: Region) -> list[dict]:
    p = region.folder / "references" / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


def _training_features(db: Database, region: Region, region_id: int) -> tuple[list, dict]:
    """All training polygons: uploaded geological maps + areas drawn in the dashboard."""
    from ..reference import load_reference_features
    feats, counts = [], {"reference_polygons": 0, "drawn_areas": 0}
    for ref in _references(region):
        f = load_reference_features(region.folder / "references" / ref["file"], ref["field"], ref.get("mapping"))
        feats += f
        counts["reference_polygons"] += len(f)
    for a in db.all("annotations", "region_id = ?", (region_id,)):
        feats.append((a["geometry"], a["class_id"]))
        counts["drawn_areas"] += 1
    return feats, counts


def _stats(region: Region) -> dict | None:
    p = region.folder / "products" / "stats.json"
    return json.loads(p.read_text()) if p.exists() else None


# ----------------------------------------------------------------------------- job handlers
def _scaled(ctx, a: float, b: float):
    return lambda p, m: ctx.progress(a + (b - a) * p, m)


def _do_acquire(ctx, region: Region, a=0.0, b=1.0):
    p = ctx.params
    region.acquire(_scaled(ctx, a, b), ctx.log, keys=p.get("tiles") or None, force=bool(p.get("force")))


def _do_train(ctx, region: Region, a=0.0, b=1.0) -> int:
    db, p = ctx.db, ctx.params
    feats, counts = _training_features(db, region, ctx.job["region_id"])
    if not feats:
        raise ValueError("No training data: upload a geological map or draw training areas on the map first")
    ctx.log(f"training data: {counts}")
    folder = ctx.root / "models" / f"region{ctx.job['region_id']}_job{ctx.job['id']}"
    name = p.get("name") or f"{region.config.name} - model {ctx.job['id']}"
    meta = region.train(folder, feats, (), p.get("algorithms") or list(ALGORITHMS), int(p.get("samples", 4000)),
                        int(p.get("epochs", 30)), name=name, progress=_scaled(ctx, a, b), log=ctx.log)
    from .app import model_summary
    summary, best = model_summary(meta)
    model_id = db.insert("models", name=name, folder=str(folder.relative_to(ctx.root)),
                         region_id=ctx.job["region_id"], uses_dem=int(meta["uses_dem"]), best_algo=best,
                         summary=summary, created_by=ctx.job["user"])
    db.update("jobs", ctx.job["id"], model_id=model_id, folder=str(folder.relative_to(ctx.root)))
    return model_id


def _do_classify(ctx, region: Region, model_id: int, a=0.0, b=1.0):
    model = ctx.db.get("models", model_id)
    if not model:
        raise ValueError("Model not found")
    region.classify(ctx.root / model["folder"], ctx.params.get("algorithm") or None,
                    int(ctx.params.get("smoothing", 3)), _scaled(ctx, a, b), ctx.log,
                    keys=ctx.params.get("tiles") or None)
    ctx.db.update("jobs", ctx.job["id"], model_id=model_id)


def _do_products(ctx, region: Region, a=0.0, b=1.0):
    from ..report import load_meta, region_report
    out = region.folder / "products"
    out.mkdir(exist_ok=True)
    region.build_mosaics(_scaled(ctx, a, a + (b - a) * 0.6))
    ctx.progress(a + (b - a) * 0.65, "Computing statistics")
    dfile = region.folder / "districts.geojson"
    districts = json.loads(dfile.read_text()) if dfile.exists() else None
    name_field = (json.loads((region.folder / "districts.json").read_text()).get("field", "name")
                  if (region.folder / "districts.json").exists() else "name")
    stats = region.statistics(districts, name_field)
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    if ctx.params.get("geojson", True):
        region.export_geojson(out / "lithology.geojson", int(ctx.params.get("min_pixels", 25)),
                              progress=_scaled(ctx, a + (b - a) * 0.7, a + (b - a) * 0.9))
    ctx.progress(a + (b - a) * 0.92, "Writing PDF report")
    model = region.state().get("model")
    region_report(region, out / "report.pdf", load_meta(model) if model else None, stats,
                  os.environ.get("ROCKMAP_ORGANISATION", ""))


def _region_for(ctx) -> Region:
    row = ctx.db.get("regions", ctx.job["region_id"])
    return Region(ctx.root / row["folder"])


@handler("region_acquire")
def job_region_acquire(ctx):
    _do_acquire(ctx, _region_for(ctx))


@handler("region_train")
def job_region_train(ctx):
    _do_train(ctx, _region_for(ctx))


@handler("region_classify")
def job_region_classify(ctx):
    _do_classify(ctx, _region_for(ctx), int(ctx.params["model_id"]))


@handler("region_products")
def job_region_products(ctx):
    _do_products(ctx, _region_for(ctx))


@handler("region_pipeline")
def job_region_pipeline(ctx):
    """Acquire -> (train) -> classify -> mosaics, statistics, GeoJSON and PDF."""
    region = _region_for(ctx)
    _do_acquire(ctx, region, 0.0, 0.45)
    model_id = ctx.params.get("model_id")
    if not model_id:
        model_id = _do_train(ctx, region, 0.45, 0.65)
    _do_classify(ctx, region, int(model_id), 0.65, 0.85)
    _do_products(ctx, region, 0.85, 1.0)


# ----------------------------------------------------------------------------- pages
@bp.route("/regions")
@requires("viewer")
def regions_page():
    rows = []
    for r in _db().all("regions"):
        try:
            rows.append({**r, "summary": _region(r).summary()})
        except (OSError, KeyError, ValueError):
            rows.append({**r, "summary": None})
    return render_template("regions.html", regions=rows)


@bp.route("/regions/new", methods=["GET", "POST"])
@requires("analyst")
def region_new():
    if request.method == "GET":
        return render_template("region_new.html", presets=PRESETS)
    f = request.form
    try:
        preset = f.get("preset") or None
        if f.get("aoi_mode") == "upload":
            up = request.files.get("aoi_file")
            if not up or not up.filename:
                raise ValueError("choose a GeoJSON boundary file")
            geom = load_geojson_geometry(json.loads(up.read()))
            preset = None
        elif f.get("aoi_mode") == "draw":
            geom = load_geojson_geometry(json.loads(f.get("aoi_geojson") or "{}"))
            if not geom.get("coordinates"):
                raise ValueError("draw a rectangle on the map first")
            preset = None
        else:
            if preset not in PRESETS:
                raise ValueError("choose a preset area")
            geom = PRESETS[preset]["geometry"]
        years = sorted({int(y) for y in f.get("years", "2023 2024 2025").replace(",", " ").split()})
        months = sorted({int(m) for m in f.get("months", "7 8 9 10").replace(",", " ").split() if 1 <= int(m) <= 12})
        source = f.get("source", "sentinel2")
        local = [s.strip() for s in f.get("local_scenes", "").splitlines() if s.strip()]
        if source == "local" and not has_role("admin"):
            raise ValueError("only administrators can use server-side files")
        for p in local + ([f["local_dem"]] if f.get("local_dem") else []):
            if not Path(p).exists():
                raise ValueError(f"file not found on server: {p}")
        name = f.get("name") or (PRESETS[preset]["name"] if preset else "New region")
        cfg = RegionConfig(name=name, aoi=geom, resolution=float(f.get("resolution") or 20),
                           tile_size=int(f.get("tile_size") or 1024), source=source, years=years, months=months,
                           max_cloud=float(f.get("max_cloud") or 30), max_scenes=int(f.get("max_scenes") or 6),
                           local_scenes=local, local_sensor=f.get("local_sensor", "sentinel2"),
                           local_dem=f.get("local_dem") or None)
        if not 5 <= cfg.resolution <= 1000 or not 128 <= cfg.tile_size <= 4096:
            raise ValueError("resolution must be 5-1000 m and tile size 128-4096 px")
        db = _db()
        rid = db.insert("regions", name=name, folder="", preset=preset, created_by=g.user["username"])
        folder = _root() / "regions" / str(rid)
        region = Region.create(folder, cfg)
        from ..region import _geom_bounds
        db.update("regions", rid, folder=str(folder.relative_to(_root())), bounds=list(_geom_bounds(geom)))
    except (ValueError, KeyError, json.JSONDecodeError) as e:
        flash(f"Could not create region: {e}", "error")
        return redirect(url_for("regions.region_new"))
    audit("region.create", f"{rid} {name} ({region.summary()['tiles']} tiles)")
    flash(f"Region created with {region.summary()['tiles']} tiles. Next: acquire imagery.", "ok")
    return redirect(url_for("regions.region_page", region_id=rid))


@bp.route("/regions/<int:region_id>")
@requires("viewer")
def region_page(region_id):
    db = _db()
    row = _region_row(region_id)
    region = _region(row)
    st = region.state()
    jobs = db.all("jobs", "region_id = ?", (region_id,), limit=15)
    active = [j for j in jobs if j["status"] in ("queued", "running")]
    feats, counts = [], {"reference_polygons": 0, "drawn_areas": db.count("annotations", "region_id = ?", (region_id,))}
    refs = _references(region)
    counts["reference_files"] = len(refs)
    mosaic = region.folder / "mosaic"
    layers = [lname for lname in LAYERS if (mosaic / f"{lname}.tif").exists()]
    products = {n: (region.folder / "products" / n).exists() for n in ("report.pdf", "lithology.geojson", "stats.json")}
    models = [m for m in db.all("models") if m["region_id"] == region_id] + \
             [m for m in db.all("models") if m["region_id"] != region_id]
    return render_template("region.html", row=row, region=region, cfg=region.config, summary=region.summary(),
                           state=st, jobs=jobs, active=active, counts=counts, refs=refs, layers=layers,
                           products=products, models=models, stats=_stats(region),
                           districts=(region.folder / "districts.geojson").exists(),
                           version=st.get("mosaic_version", 0), algorithms=ALGORITHMS,
                           classes=[{"id": c.id, "name": c.name, "color": c.color} for c in ROCK_CLASSES],
                           focus_job=request.args.get("job", type=int))


@bp.post("/regions/<int:region_id>/run")
@requires("analyst")
def region_run(region_id):
    row = _region_row(region_id)
    region = _region(row)
    stage = request.form.get("stage")
    f = request.form
    tiles = [t for t in (f.get("tiles") or "").split(",") if t in set(region.tile_keys)] or None
    params: dict = {"tiles": tiles}
    kind = {"acquire": "region_acquire", "train": "region_train", "classify": "region_classify",
            "products": "region_products", "pipeline": "region_pipeline"}.get(stage)
    if not kind:
        abort(400)
    if stage in ("acquire", "pipeline"):
        params["force"] = bool(f.get("force"))
    if stage in ("train", "pipeline"):
        params.update(algorithms=[a for a in f.getlist("algorithms") if a in ALGORITHMS] or list(ALGORITHMS),
                      epochs=min(200, max(1, int(f.get("epochs") or 30))),
                      samples=min(100_000, max(200, int(f.get("samples") or 4000))), name=f.get("model_name") or None)
    if stage in ("classify", "pipeline"):
        params.update(algorithm=f.get("algorithm") if f.get("algorithm") in ALGORITHMS else None,
                      smoothing=int(f.get("smoothing") or 3))
        if f.get("model_id"):
            params["model_id"] = int(f["model_id"])
        elif stage == "classify":
            flash("Choose a model to classify with.", "error")
            return redirect(url_for("regions.region_page", region_id=region_id))
    if stage == "products":
        params.update(geojson=bool(f.get("geojson")), min_pixels=int(f.get("min_pixels") or 25))
    job_id = current_app.extensions["rockmap_submit"](kind, params, region_id=region_id)
    flash(f"Job #{job_id} queued.", "ok")
    return redirect(url_for("regions.region_page", region_id=region_id, job=job_id))


@bp.post("/regions/<int:region_id>/reference")
@requires("analyst")
def region_reference(region_id):
    region = _region(_region_row(region_id))
    up = request.files.get("file")
    if not up or not up.filename:
        flash("Choose a GeoJSON geological map.", "error")
        return redirect(url_for("regions.region_page", region_id=region_id))
    try:
        data = json.loads(up.read())
        mapping_txt = request.form.get("mapping", "").strip()
        mapping = json.loads(mapping_txt) if mapping_txt else None
        field = request.form.get("field") or "class_id"
        refdir = region.folder / "references"
        refdir.mkdir(exist_ok=True)
        refs = _references(region)
        fname = f"ref{len(refs) + 1}_{secure_filename(up.filename) or 'map.geojson'}"
        (refdir / fname).write_text(json.dumps(data))
        from ..reference import load_reference_features
        n = len(load_reference_features(refdir / fname, field, mapping))
        if n == 0:
            (refdir / fname).unlink()
            raise ValueError(f"no polygon could be mapped to a rock class (field '{field}')")
        refs.append({"file": fname, "name": up.filename, "field": field, "mapping": mapping, "polygons": n,
                     "by": g.user["username"]})
        (refdir / "index.json").write_text(json.dumps(refs, indent=2))
        audit("region.reference", f"{region_id} {up.filename} ({n} polygons)")
        flash(f"Reference map added: {n} usable polygons.", "ok")
    except (ValueError, json.JSONDecodeError, KeyError) as e:
        flash(f"Reference map rejected: {e}", "error")
    return redirect(url_for("regions.region_page", region_id=region_id))


@bp.post("/regions/<int:region_id>/reference/<int:idx>/delete")
@requires("analyst")
def region_reference_delete(region_id, idx):
    region = _region(_region_row(region_id))
    refs = _references(region)
    if 0 <= idx < len(refs):
        ref = refs.pop(idx)
        (region.folder / "references" / ref["file"]).unlink(missing_ok=True)
        (region.folder / "references" / "index.json").write_text(json.dumps(refs, indent=2))
        audit("region.reference.delete", f"{region_id} {ref['name']}")
    return redirect(url_for("regions.region_page", region_id=region_id))


@bp.post("/regions/<int:region_id>/districts")
@requires("analyst")
def region_districts(region_id):
    region = _region(_region_row(region_id))
    up = request.files.get("file")
    try:
        data = json.loads(up.read()) if up and up.filename else None
        if not data or data.get("type") != "FeatureCollection":
            raise ValueError("upload a GeoJSON FeatureCollection of district polygons")
        (region.folder / "districts.geojson").write_text(json.dumps(data))
        (region.folder / "districts.json").write_text(json.dumps({"field": request.form.get("field") or "name"}))
        audit("region.districts", f"{region_id} ({len(data['features'])} districts)")
        flash(f"{len(data['features'])} districts saved. Run 'Build products' to compute district statistics.", "ok")
    except (ValueError, json.JSONDecodeError) as e:
        flash(f"Districts rejected: {e}", "error")
    return redirect(url_for("regions.region_page", region_id=region_id))


@bp.post("/regions/<int:region_id>/delete")
@requires("admin")
def region_delete(region_id):
    db = _db()
    row = _region_row(region_id)
    if db.count("jobs", "region_id = ? AND status IN ('queued','running')", (region_id,)):
        flash("Cancel the running jobs of this region first.", "error")
        return redirect(url_for("regions.region_page", region_id=region_id))
    shutil.rmtree(_root() / row["folder"], ignore_errors=True)
    db.execute("DELETE FROM annotations WHERE region_id = ?", (region_id,))
    db.delete("regions", region_id)
    audit("region.delete", row["name"])
    flash("Region deleted.", "ok")
    return redirect(url_for("regions.regions_page"))


@bp.route("/regions/<int:region_id>/stats.csv")
@requires("viewer")
def region_stats_csv(region_id):
    stats = _stats(_region(_region_row(region_id))) or abort(404)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["area", "class_id", "class", "area_km2", "percent_of_rock"])
    for s in stats["region"]:
        w.writerow(["region", s["id"], s["name"], round(s["area_km2"], 4),
                    "" if s["percent"] is None else round(s["percent"], 3)])
    for d in stats.get("districts", []):
        for s in d["stats"]:
            w.writerow([d["name"], s["id"], s["name"], round(s["area_km2"], 4),
                        "" if s["percent"] is None else round(s["percent"], 3)])
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=region{region_id}_stats.csv"})


# ----------------------------------------------------------------------------- map tiles
@bp.route("/tiles/<int:region_id>/<layer>/<int:z>/<int:x>/<int:y>.png")
@requires("viewer")
def tile(region_id, layer, z, x, y):
    from ..tiles import cached_tile, empty_png
    if layer not in LAYERS or z > 20:
        abort(404)
    region = _region(_region_row(region_id))
    mosaic = region.folder / "mosaic" / f"{layer}.tif"
    data = cached_tile(region.folder / "cache" / "tiles", mosaic, layer, z, x, y) if mosaic.exists() else empty_png()
    resp = Response(data, mimetype="image/png")
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


# ----------------------------------------------------------------------------- JSON API
@bp.route("/api/regions")
@requires("viewer")
def api_regions():
    out = []
    for r in _db().all("regions"):
        try:
            out.append({**r, "summary": _region(r).summary()})
        except (OSError, KeyError, ValueError):
            continue
    return jsonify(out)


@bp.route("/api/regions/<int:region_id>")
@requires("viewer")
def api_region(region_id):
    row = _region_row(region_id)
    region = _region(row)
    st = region.state()
    return jsonify({**row, "config": region.config.__dict__, "summary": region.summary(),
                    "model": st.get("model"), "algorithm": st.get("algorithm"),
                    "mosaic_version": st.get("mosaic_version")})


@bp.route("/api/regions/<int:region_id>/tiles")
@requires("viewer")
def api_region_tiles(region_id):
    region = _region(_region_row(region_id))
    cache = region.folder / "cache" / "tilegrid.json"
    state_mtime = (region.folder / "state.json").stat().st_mtime
    if cache.exists() and cache.stat().st_mtime >= state_mtime:
        return Response(cache.read_text(), mimetype="application/json")
    data = region.tile_geojson()
    data["aoi"] = region.config.aoi
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return jsonify(data)


@bp.route("/api/regions/<int:region_id>/query")
@requires("viewer")
def api_region_query(region_id):
    region = _region(_region_row(region_id))
    try:
        lat, lon = float(request.args["lat"]), float(request.args["lon"])
    except (KeyError, ValueError):
        abort(400)
    return jsonify(region.query(lon, lat))


@bp.route("/api/regions/<int:region_id>/stats")
@requires("viewer")
def api_region_stats(region_id):
    return jsonify(_stats(_region(_region_row(region_id))) or {})


@bp.route("/api/regions/<int:region_id>/annotations", methods=["GET", "POST"])
@requires("viewer")
def api_annotations(region_id):
    db = _db()
    _region_row(region_id)
    if request.method == "POST":
        if not has_role("analyst"):
            return jsonify(error="'analyst' role required"), 403
        d = request.get_json(silent=True) or {}
        geom = d.get("geometry") or {}
        try:
            cid = int(d.get("class_id"))
        except (TypeError, ValueError):
            cid = -1
        if cid not in CLASS_IDS or geom.get("type") not in ("Polygon", "MultiPolygon"):
            return jsonify(error="need a Polygon geometry and a rock class_id 1-7"), 400
        ring = geom["coordinates"][0] if geom["type"] == "Polygon" else geom["coordinates"][0][0]
        if len(ring) < 4:
            return jsonify(error="polygon needs at least 3 vertices"), 400
        aid = db.insert("annotations", region_id=region_id, class_id=cid, geometry=geom,
                        note=(d.get("note") or "")[:500], author=g.user["username"])
        audit("annotation.create", f"region {region_id} class {cid}")
        return jsonify(id=aid), 201
    return jsonify({"type": "FeatureCollection", "features": [
        {"type": "Feature", "id": a["id"], "geometry": a["geometry"],
         "properties": {"id": a["id"], "class_id": a["class_id"], "class": ALL_CLASSES[a["class_id"]].name,
                        "color": ALL_CLASSES[a["class_id"]].color, "note": a["note"], "author": a["author"],
                        "created": a["created"]}}
        for a in db.all("annotations", "region_id = ?", (region_id,))]})


@bp.route("/api/regions/<int:region_id>/annotations/<int:aid>", methods=["DELETE"])
@requires("analyst")
def api_annotation_delete(region_id, aid):
    db = _db()
    a = db.get("annotations", aid)
    if not a or a["region_id"] != region_id:
        abort(404)
    if a["author"] != g.user["username"] and not has_role("admin"):
        return jsonify(error="only the author or an admin can delete this training area"), 403
    db.delete("annotations", aid)
    audit("annotation.delete", f"region {region_id} #{aid}")
    return jsonify(ok=True)


@bp.post("/api/regions/<int:region_id>/jobs")
@requires("analyst")
def api_region_job(region_id):
    """Start a region job: {"stage": "acquire|train|classify|products|pipeline", ...params}."""
    _region_row(region_id)
    d = request.get_json(silent=True) or {}
    kind = {"acquire": "region_acquire", "train": "region_train", "classify": "region_classify",
            "products": "region_products", "pipeline": "region_pipeline"}.get(d.pop("stage", None))
    if not kind:
        return jsonify(error="stage must be acquire, train, classify, products or pipeline"), 400
    if kind == "region_classify" and not d.get("model_id"):
        return jsonify(error="model_id required"), 400
    job_id = current_app.extensions["rockmap_submit"](kind, d, region_id=region_id)
    return jsonify(job_id=job_id), 202


__all__ = ["bp", "LAYERS", "CLOUD_CLASS", "ALGORITHM_LABELS"]
