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
from ..acquisition import recent_years
from ..presets import PRESETS
from ..region import Region, RegionConfig, load_geojson_geometry
from .auth import audit, has_role, requires
from .db import Database
from .jobs import handler

bp = Blueprint("regions", __name__)
LAYERS = ("rgb", "falsecolor", "hillshade", "surface", "lithology", "confidence", "alteration", "hazard", "clusters",
          "gems", "gem_marble", "gem_pegmatite", "gem_contact", "gem_ultramafic", "gem_ml",
          "minerals", "min_iron", "min_copper", "min_vein", "min_ml", "lineaments", "hyper", "hyper_iron", "vhr")
STAGES = {"acquire": "region_acquire", "analyze": "region_analyze", "gems": "region_gems",
          "minerals": "region_minerals", "train": "region_train",
          "classify": "region_classify", "products": "region_products", "pipeline": "region_pipeline"}


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
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []


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
    from .field import observation_squares
    obs = observation_squares(db, region_id)
    feats += obs
    counts["field_observations"] = len(obs)
    return feats, counts


def _stats(region: Region) -> dict | None:
    p = region.folder / "products" / "stats.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


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
    model_id = db.insert("models", name=name, folder=folder.relative_to(ctx.root).as_posix(),
                         region_id=ctx.job["region_id"], uses_dem=int(meta["uses_dem"]), best_algo=best,
                         summary=summary, created_by=ctx.job["user"])
    db.update("jobs", ctx.job["id"], model_id=model_id, folder=folder.relative_to(ctx.root).as_posix())
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
    districts = json.loads(dfile.read_text(encoding="utf-8")) if dfile.exists() else None
    name_field = (json.loads((region.folder / "districts.json").read_text(encoding="utf-8")).get("field", "name")
                  if (region.folder / "districts.json").exists() else "name")
    stats = region.statistics(districts, name_field)
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if ctx.params.get("geojson", True):
        region.export_geojson(out / "lithology.geojson", int(ctx.params.get("min_pixels", 25)),
                              progress=_scaled(ctx, a + (b - a) * 0.7, a + (b - a) * 0.9))
    ctx.progress(a + (b - a) * 0.92, "Writing PDF report")
    model = region.state().get("model")
    obs = ctx.db.all("observations", "region_id = ?", (ctx.job["region_id"],))
    validation = region.field_validation(obs) if obs and stats.get("classified_km2") else None
    region_report(region, out / "report.pdf", load_meta(model) if model else None, stats,
                  os.environ.get("ROCKMAP_ORGANISATION", ""), analytics=region.analytics(), validation=validation,
                  gems=region.gems(), minerals=region.minerals())
    ctx.progress(a + (b - a) * 0.97, "Writing GeoPackage")
    region.export_geopackage(extra={"field_observations": observations_layer(ctx.db, ctx.job["region_id"])})


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


@handler("region_analyze")
def job_region_analyze(ctx):
    _region_for(ctx).analyze(int(ctx.params.get("n_clusters", 10)), ctx.progress, ctx.log)


def gem_occurrences(db: Database, region: Region, region_id: int) -> list[dict]:
    """Known gem localities: uploaded lists + field-app finds that name a gem."""
    from ..gems import GEM_TYPES
    occ = list(region.gem_occurrences())
    for o in db.all("observations", "region_id = ? AND gem IS NOT NULL AND gem != ''", (region_id,)):
        occ.append({"lat": o["lat"], "lon": o["lon"], "gem": o["gem"], "name": f"field #{o['id']}",
                    "model": GEM_TYPES.get(o["gem"]), "source": "field"})
    return occ


def _do_gems(ctx, region: Region, a=0.0, b=1.0):
    occ = gem_occurrences(ctx.db, region, ctx.job["region_id"])
    ctx.log(f"known gem occurrences: {len(occ)}")
    region.gem_analysis(occ, _scaled(ctx, a, b), ctx.log)


def observations_layer(db: Database, region_id: int) -> dict:
    """Field observations of a region as a GeoJSON layer (for the GeoPackage)."""
    from ..config import ALL_CLASSES
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
         "properties": {"id": o["id"], "rock": ALL_CLASSES[o["class_id"]].name if o["class_id"] in ALL_CLASSES else
                        str(o["class_id"]), "certainty": o["certainty"], "gem": o.get("gem"),
                        "commodity": o.get("commodity"), "note": o["note"], "author": o["author"],
                        "observed_at": o["observed_at"] or o["created"]}}
        for o in db.all("observations", "region_id = ?", (region_id,))]}


def mineral_occurrences(db: Database, region: Region, region_id: int) -> list[dict]:
    """Known mineral occurrences: uploaded lists + field-app observations that name a commodity."""
    from ..minerals import COMMODITY_TYPES
    occ = list(region.mineral_occurrences())
    for o in db.all("observations", "region_id = ? AND commodity IS NOT NULL AND commodity != ''", (region_id,)):
        occ.append({"lat": o["lat"], "lon": o["lon"], "commodity": o["commodity"], "name": f"field #{o['id']}",
                    "model": COMMODITY_TYPES.get(o["commodity"]), "source": "field"})
    return occ


def _do_minerals(ctx, region: Region, a=0.0, b=1.0):
    occ = mineral_occurrences(ctx.db, region, ctx.job["region_id"])
    ctx.log(f"known mineral occurrences: {len(occ)}")
    res = region.mineral_analysis(occ, _scaled(ctx, a, b), ctx.log)
    ctx.log(f"lineaments: {res['lineaments']['segments']} ({res['lineaments']['total_km']} km); "
            f"targets: {res['targets_total']}")


@handler("region_minerals")
def job_region_minerals(ctx):
    region = _region_for(ctx)
    _do_minerals(ctx, region, 0.0, 0.85)
    region.build_mosaics(_scaled(ctx, 0.85, 1.0))


@handler("region_aster")
def job_region_aster(ctx):
    region = _region_for(ctx)
    res = region.import_aster(ctx.params["paths"], _scaled(ctx, 0.0, 0.3))
    ctx.log(f"ASTER Quartz Index imported: {res['coverage'] * 100:.0f} % of the region covered")
    _do_minerals(ctx, region, 0.3, 0.9)
    region.build_mosaics(_scaled(ctx, 0.9, 1.0))


@handler("region_hyperspectral")
def job_region_hyperspectral(ctx):
    region = _region_for(ctx)
    lib = Path(ctx.params["library"]).read_bytes() if ctx.params.get("library") else None
    info = region.import_hyperspectral(ctx.params["paths"], ctx.params.get("wavelengths"), lib,
                                       _scaled(ctx, 0.0, 0.5), ctx.log)
    ctx.log(f"hyperspectral coverage: {info['covered_km2']:.1f} km2 of the region")
    _do_minerals(ctx, region, 0.5, 0.9)
    region.build_mosaics(_scaled(ctx, 0.9, 1.0))


@handler("region_vhr")
def job_region_vhr(ctx):
    region = _region_for(ctx)
    info = region.import_vhr(ctx.params["path"], progress=_scaled(ctx, 0.0, 1.0))
    ctx.log(f"very-high-resolution layer ready: {info['resolution_m']} m pixels, bands {info['bands']}")


@handler("region_gems")
def job_region_gems(ctx):
    region = _region_for(ctx)
    _do_gems(ctx, region, 0.0, 0.85)
    region.build_mosaics(_scaled(ctx, 0.85, 1.0))


@handler("region_label_units")
def job_region_label_units(ctx):
    region = _region_for(ctx)
    region.label_clusters(ctx.params["mapping"], _scaled(ctx, 0, 0.5), ctx.log)
    _do_products(ctx, region, 0.5, 1.0)


@handler("region_pipeline")
def job_region_pipeline(ctx):
    """Acquire -> analytics -> gems -> lineaments & minerals -> (train) -> classify -> products.

    Without a model and without training data the lithology step is skipped, but imagery,
    surface cover, alteration targets, landslide susceptibility and spectral units are produced.
    """
    region = _region_for(ctx)
    _do_acquire(ctx, region, 0.0, 0.4)
    region.analyze(int(ctx.params.get("n_clusters", 10)), _scaled(ctx, 0.4, 0.5), ctx.log)
    _do_gems(ctx, region, 0.5, 0.55)
    _do_minerals(ctx, region, 0.55, 0.62)
    model_id = ctx.params.get("model_id")
    if not model_id:
        feats, _counts = _training_features(ctx.db, region, ctx.job["region_id"])
        if feats:
            model_id = _do_train(ctx, region, 0.62, 0.72)
        else:
            from .trained import current_real_model
            row = current_real_model(ctx.db)
            if row:
                model_id = row["id"]
                ctx.log(f"no model chosen and no training data: using the current trained model {row['version']}")
            else:
                ctx.log("no model and no training data: skipping lithology classification")
    if model_id:
        _do_classify(ctx, region, int(model_id), 0.72, 0.85)
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
        return render_template("region_new.html", presets=PRESETS,
                               default_years=" ".join(map(str, recent_years())))
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
        years = sorted({int(y) for y in (f.get("years") or " ".join(map(str, recent_years()))).replace(",", " ").split()})
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
        db.update("regions", rid, folder=folder.relative_to(_root()).as_posix(), bounds=list(_geom_bounds(geom)))
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
    from ..analytics import HAZARD_CLASSES, insights
    analytics = region.analytics()
    stats = _stats(region)
    obs = db.all("observations", "region_id = ?", (region_id,))
    counts["field_observations"] = len(obs)
    validation = region.field_validation(obs) if obs and summary_classified(region) else None
    products.update({n: (region.folder / "products" / n).exists() for n in ("targets.csv", "targets.geojson")})
    share_url = url_for("regions.share_page", token=row["share_token"], _external=True) if row.get("share_token") else None
    from ..gems import GEM_MODELS, GEM_NAMES
    gems = region.gems()
    products.update({n: (region.folder / "products" / n).exists() for n in ("gem_targets.csv", "gem_targets.geojson")})
    from ..hyperspectral import MINERALS as HYPER_MINERALS
    from ..minerals import COMMODITY_NAMES, MINERAL_MODELS
    minerals = region.minerals()
    products.update({n: (region.folder / "products" / n).exists()
                     for n in ("mineral_targets.csv", "mineral_targets.geojson", "lineaments.geojson")})
    return render_template("region.html", gems=gems, gem_models=GEM_MODELS, gem_names=GEM_NAMES,
                           minerals=minerals, mineral_models=MINERAL_MODELS, commodity_names=COMMODITY_NAMES,
                           hyper=region.hyperspectral(), hyper_minerals=HYPER_MINERALS,
                           min_occ=mineral_occurrences(db, region, region_id),
                           gem_occ=gem_occurrences(db, region, region_id), row=row, region=region, cfg=region.config, summary=region.summary(),
                           state=st, jobs=jobs, active=active, counts=counts, refs=refs, layers=layers,
                           products=products, models=models, stats=stats, analytics=analytics,
                           insights=insights(stats, analytics, gems, minerals), hazard_classes=HAZARD_CLASSES,
                           validation=validation, share_url=share_url,
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
    kind = STAGES.get(stage)
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
    if stage in ("analyze", "pipeline"):
        params["n_clusters"] = min(16, max(2, int(f.get("n_clusters") or 10)))
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
        (refdir / fname).write_text(json.dumps(data), encoding="utf-8")
        from ..reference import load_reference_features
        n = len(load_reference_features(refdir / fname, field, mapping))
        if n == 0:
            (refdir / fname).unlink()
            raise ValueError(f"no polygon could be mapped to a rock class (field '{field}')")
        refs.append({"file": fname, "name": up.filename, "field": field, "mapping": mapping, "polygons": n,
                     "by": g.user["username"]})
        (refdir / "index.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")
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
        (region.folder / "references" / "index.json").write_text(json.dumps(refs, indent=2), encoding="utf-8")
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
        (region.folder / "districts.geojson").write_text(json.dumps(data), encoding="utf-8")
        (region.folder / "districts.json").write_text(json.dumps({"field": request.form.get("field") or "name"}), encoding="utf-8")
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


def summary_classified(region: Region) -> bool:
    return region.summary()["classified"] > 0


@bp.post("/regions/<int:region_id>/units")
@requires("analyst")
def region_units(region_id):
    """Geologist names the spectral units -> lithology map without a trained model."""
    region = _region(_region_row(region_id))
    info = region.analytics() or abort(400)
    mapping = {}
    for c in info["clusters"]:
        v = request.form.get(f"unit_{c['id']}")
        if v and v.isdigit() and int(v) in CLASS_IDS:
            mapping[c["id"]] = int(v)
    if not mapping:
        flash("Assign at least one spectral unit to a rock class.", "error")
        return redirect(url_for("regions.region_page", region_id=region_id))
    job_id = current_app.extensions["rockmap_submit"]("region_label_units", {"mapping": mapping}, region_id=region_id)
    return redirect(url_for("regions.region_page", region_id=region_id, job=job_id))


@bp.post("/regions/<int:region_id>/share")
@requires("analyst")
def region_share(region_id):
    import secrets
    _region_row(region_id)
    token = secrets.token_urlsafe(18) if request.form.get("action") == "create" else None
    _db().update("regions", region_id, share_token=token)
    audit("region.share." + ("create" if token else "revoke"), str(region_id))
    flash("Public read-only link created." if token else "Public link revoked.", "ok")
    return redirect(url_for("regions.region_page", region_id=region_id))


@bp.route("/regions/<int:region_id>/targets.<fmt>")
@requires("viewer")
def region_targets_file(region_id, fmt):
    if fmt not in ("csv", "geojson"):
        abort(404)
    region = _region(_region_row(region_id))
    p = region.folder / "products" / f"targets.{fmt}"
    if not p.exists():
        abort(404)
    return Response(p.read_bytes(), mimetype="text/csv" if fmt == "csv" else "application/geo+json",
                    headers={"Content-Disposition": f"attachment; filename=region{region_id}_targets.{fmt}"})


QML_LAYERS = ("lithology", "surface", "hazard", "clusters")


@bp.route("/regions/<int:region_id>/style/<layer>.qml")
@requires("viewer")
def region_style(region_id, layer):
    """QGIS style file (paletted renderer) for the downloaded GeoTIFF mosaics."""
    from ..analytics import CLUSTER_PALETTE, HAZARD_CLASSES
    if layer not in QML_LAYERS:
        abort(404)
    if layer in ("lithology", "surface"):
        entries = [(c.id, c.color, c.name) for c in ALL_CLASSES.values() if c.id != CLOUD_CLASS]
        if layer == "surface":
            entries = [(1, "#cdaa7d", "Bare rock / soil")] + [e for e in entries if e[0] >= 250]
    elif layer == "hazard":
        entries = [(k, c, f"{n} susceptibility") for k, (n, c) in HAZARD_CLASSES.items()]
    else:
        info = _region(_region_row(region_id)).analytics() or {"clusters": []}
        entries = [(c["id"], CLUSTER_PALETTE[(c["id"] - 1) % len(CLUSTER_PALETTE)],
                    f"Unit {c['id']}" + (f" - {ALL_CLASSES[c['class_id']].name}" if c.get("class_id") else ""))
                   for c in info["clusters"]]
    from xml.sax.saxutils import quoteattr
    items = "\n".join(f'        <paletteEntry value="{v}" color="{c}" alpha="255" label={quoteattr(n)}/>'
                      for v, c, n in entries)
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28" styleCategories="Symbology">
  <pipe>
    <rasterrenderer type="paletted" band="1" opacity="1" alphaBand="-1" nodataColor="">
      <colorPalette>
{items}
      </colorPalette>
    </rasterrenderer>
  </pipe>
</qgis>
"""
    return Response(qml, mimetype="application/xml",
                    headers={"Content-Disposition": f"attachment; filename={layer}.qml"})


# ----------------------------------------------------------------------------- public share pages
def _shared(token: str) -> dict:
    row = _db().one("regions", "share_token = ?", (token,)) if token else None
    if not row:
        abort(404)
    return row


@bp.route("/share/<token>")
def share_page(token):
    from ..analytics import HAZARD_CLASSES, insights
    row = _shared(token)
    region = _region(row)
    st = region.state()
    analytics = region.analytics()
    stats = _stats(region)
    layers = [lname for lname in LAYERS if (region.folder / "mosaic" / f"{lname}.tif").exists()]
    from ..gems import GEM_MODELS
    gems = region.gems()
    return render_template("share.html", row=row, cfg=region.config, token=token, layers=layers,
                           version=st.get("mosaic_version", 0), stats=stats, analytics=analytics,
                           gems=gems, gem_models=GEM_MODELS,
                           insights=insights(stats, analytics, gems), hazard_classes=HAZARD_CLASSES,
                           classes=[{"id": c.id, "name": c.name, "color": c.color} for c in ROCK_CLASSES])


@bp.route("/share/<token>/tiles/<layer>/<int:z>/<int:x>/<int:y>.png")
def share_tile(token, layer, z, x, y):
    return _tile_response(_region(_shared(token)), layer, z, x, y)


@bp.route("/share/<token>/api/tiles")
def share_tiles_api(token):
    return _tilegrid(_region(_shared(token)))


@bp.route("/share/<token>/api/query")
def share_query(token):
    return _query(_region(_shared(token)))


@bp.route("/share/<token>/api/targets")
def share_targets(token):
    return _targets(_region(_shared(token)))


@bp.route("/share/<token>/api/gem-targets")
def share_gem_targets(token):
    return _targets(_region(_shared(token)), "gem_targets.geojson")


# ----------------------------------------------------------------------------- map tiles
@bp.route("/tiles/<int:region_id>/<layer>/<int:z>/<int:x>/<int:y>.png")
@requires("viewer")
def tile(region_id, layer, z, x, y):
    return _tile_response(_region(_region_row(region_id)), layer, z, x, y)


def _tile_response(region: Region, layer: str, z: int, x: int, y: int):
    from ..tiles import cached_tile, empty_png
    if layer not in LAYERS or z > 20:
        abort(404)
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
    return _tilegrid(_region(_region_row(region_id)))


def _tilegrid(region: Region):
    cache = region.folder / "cache" / "tilegrid.json"
    state_mtime = (region.folder / "state.json").stat().st_mtime
    if cache.exists() and cache.stat().st_mtime >= state_mtime:
        return Response(cache.read_text(encoding="utf-8"), mimetype="application/json")
    data = region.tile_geojson()
    data["aoi"] = region.config.aoi
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data), encoding="utf-8")
    return jsonify(data)


@bp.route("/api/regions/<int:region_id>/query")
@requires("viewer")
def api_region_query(region_id):
    return _query(_region(_region_row(region_id)))


@bp.route("/api/regions/<int:region_id>/targets")
@requires("viewer")
def api_region_targets(region_id):
    return _targets(_region(_region_row(region_id)))


@bp.route("/api/regions/<int:region_id>/gem-targets")
@requires("viewer")
def api_region_gem_targets(region_id):
    return _targets(_region(_region_row(region_id)), "gem_targets.geojson")


@bp.route("/api/regions/<int:region_id>/gem-occurrences")
@requires("viewer")
def api_region_gem_occurrences(region_id):
    region = _region(_region_row(region_id))
    occ = gem_occurrences(_db(), region, region_id)
    return jsonify({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
         "properties": {"gem": o.get("gem"), "name": o.get("name"), "model": o.get("model"),
                        "source": o.get("source", "upload")}} for o in occ]})


@bp.post("/regions/<int:region_id>/gem-occurrences")
@requires("analyst")
def region_gem_occurrences(region_id):
    """Upload known gem localities (CSV lat,lon,gem,name or GeoJSON points)."""
    from ..gems import parse_occurrences
    region = _region(_region_row(region_id))
    if request.form.get("action") == "clear":
        region.set_gem_occurrences([])
        audit("gems.occurrences.clear", str(region_id))
        flash("Known gem localities removed.", "ok")
        return redirect(url_for("regions.region_page", region_id=region_id))
    up = request.files.get("file")
    try:
        if not up or not up.filename:
            raise ValueError("choose a CSV or GeoJSON file")
        occ = parse_occurrences(up.read(), up.filename)
        if not occ:
            raise ValueError("no points found (CSV needs lat, lon and gem columns)")
        existing = region.gem_occurrences() if request.form.get("mode") == "append" else []
        region.set_gem_occurrences(existing + occ)
        unknown = sorted({o["gem"] for o in occ if not o["model"]})
        audit("gems.occurrences.upload", f"{region_id}: {len(occ)} points")
        flash(f"{len(occ)} known gem localities saved." + (f" Unrecognised gem names (kept, not used for "
              f"per-model validation): {', '.join(unknown)}" if unknown else "") +
              " Run 'Gem prospectivity' to validate the maps with them.", "ok")
    except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as e:
        flash(f"Localities rejected: {e}", "error")
    return redirect(url_for("regions.region_page", region_id=region_id))


@bp.route("/api/regions/<int:region_id>/mineral-targets")
@requires("viewer")
def api_region_mineral_targets(region_id):
    return _targets(_region(_region_row(region_id)), "mineral_targets.geojson")


@bp.route("/api/regions/<int:region_id>/lineaments")
@requires("viewer")
def api_region_lineaments(region_id):
    return _targets(_region(_region_row(region_id)), "lineaments.geojson")


@bp.route("/api/regions/<int:region_id>/mineral-occurrences")
@requires("viewer")
def api_region_mineral_occurrences(region_id):
    region = _region(_region_row(region_id))
    return jsonify({"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
         "properties": {"commodity": o.get("commodity"), "name": o.get("name"), "model": o.get("model"),
                        "source": o.get("source", "upload")}} for o in mineral_occurrences(_db(), region, region_id)]})


@bp.post("/regions/<int:region_id>/mineral-occurrences")
@requires("analyst")
def region_mineral_occurrences(region_id):
    """Upload known mineral occurrences (CSV lat,lon,commodity,name or GeoJSON points)."""
    from ..minerals import parse_occurrences
    region = _region(_region_row(region_id))
    if request.form.get("action") == "clear":
        region.set_mineral_occurrences([])
        audit("minerals.occurrences.clear", str(region_id))
        flash("Known mineral occurrences removed.", "ok")
        return redirect(url_for("regions.region_page", region_id=region_id))
    up = request.files.get("file")
    try:
        if not up or not up.filename:
            raise ValueError("choose a CSV or GeoJSON file")
        occ = parse_occurrences(up.read(), up.filename)
        if not occ:
            raise ValueError("no points found (CSV needs lat, lon and commodity columns)")
        existing = region.mineral_occurrences() if request.form.get("mode") == "append" else []
        region.set_mineral_occurrences(existing + occ)
        unknown = sorted({o["commodity"] for o in occ if not o["model"]})
        audit("minerals.occurrences.upload", f"{region_id}: {len(occ)} points")
        flash(f"{len(occ)} known mineral occurrences saved." + (f" Commodities without a model (kept, not "
              f"validated): {', '.join(unknown)}" if unknown else "") +
              " Run 'Lineaments & mineral prospectivity' to validate the maps with them.", "ok")
    except (ValueError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as e:
        flash(f"Occurrences rejected: {e}", "error")
    return redirect(url_for("regions.region_page", region_id=region_id))


@bp.post("/regions/<int:region_id>/aster")
@requires("analyst")
def region_aster(region_id):
    """Upload ASTER thermal emissivity GeoTIFF(s) (AST_05 bands 10-14) -> Quartz Index -> vein model."""
    row = _region_row(region_id)
    region = _region(row)
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("Choose one or more ASTER emissivity GeoTIFFs (bands 10-14).", "error")
        return redirect(url_for("regions.region_page", region_id=region_id))
    folder = region.folder / "aster"
    folder.mkdir(exist_ok=True)
    paths = []
    for f in files:
        name = secure_filename(f.filename) or "aster.tif"
        if not name.lower().endswith((".tif", ".tiff")):
            flash(f"{name}: only GeoTIFF is accepted (convert HDF with QGIS or gdal_translate).", "error")
            return redirect(url_for("regions.region_page", region_id=region_id))
        f.save(folder / name)
        paths.append(str(folder / name))
    jid = current_app.extensions["rockmap_submit"]("region_aster", {"paths": paths}, region_id=region_id)
    audit("minerals.aster.upload", f"{region_id}: {len(paths)} files")
    flash(f"{len(paths)} ASTER file(s) uploaded; Quartz Index and mineral maps are being computed.", "ok")
    return redirect(url_for("regions.region_page", region_id=region_id, job=jid))


@bp.post("/regions/<int:region_id>/hyperspectral")
@requires("analyst")
def region_hyperspectral(region_id):
    """Upload hyperspectral scene(s) (GeoTIFF reflectance) + optional wavelengths (.hdr/.txt) and
    spectral library (.csv) -> alteration mineral maps -> copper / iron models recomputed."""
    from ..hyperspectral import parse_wavelengths
    region = _region(_region_row(region_id))
    back = redirect(url_for("regions.region_page", region_id=region_id))
    scenes = [f for f in request.files.getlist("scenes") if f and f.filename]
    if not scenes:
        flash("Choose one or more hyperspectral GeoTIFFs (EnMAP / PRISMA surface reflectance).", "error")
        return back
    folder = region.folder / "hyperspectral" / "input"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in scenes:
        name = secure_filename(f.filename) or "scene.tif"
        if not name.lower().endswith((".tif", ".tiff")):
            flash(f"{name}: only GeoTIFF is accepted (export EnMAP / PRISMA to GeoTIFF first).", "error")
            return back
        f.save(folder / name)
        paths.append(str(folder / name))
    params: dict = {"paths": paths}
    wl = request.files.get("wavelengths")
    if wl and wl.filename:
        try:
            params["wavelengths"] = [float(v) for v in parse_wavelengths(wl.read().decode("utf-8", "ignore"))]
        except ValueError as e:
            flash(f"Wavelengths rejected: {e}", "error")
            return back
    lib = request.files.get("library")
    if lib and lib.filename:
        lp = folder / (secure_filename(lib.filename) or "library.csv")
        lib.save(lp)
        params["library"] = str(lp)
    jid = current_app.extensions["rockmap_submit"]("region_hyperspectral", params, region_id=region_id)
    audit("minerals.hyperspectral.upload", f"{region_id}: {len(paths)} scenes")
    flash(f"{len(paths)} hyperspectral scene(s) uploaded; mineral maps are being computed.", "ok")
    return redirect(url_for("regions.region_page", region_id=region_id, job=jid))


@bp.post("/regions/<int:region_id>/vhr")
@requires("analyst")
def region_vhr(region_id):
    """Upload very-high-resolution imagery (GeoTIFF) as a map layer."""
    region = _region(_region_row(region_id))
    f = request.files.get("file")
    if not f or not f.filename or not f.filename.lower().endswith((".tif", ".tiff")):
        flash("Choose a GeoTIFF (WorldView-3, Pleiades, SuperView ... ortho-image).", "error")
        return redirect(url_for("regions.region_page", region_id=region_id))
    folder = region.folder / "vhr"
    folder.mkdir(exist_ok=True)
    path = folder / (secure_filename(f.filename) or "vhr.tif")
    f.save(path)
    jid = current_app.extensions["rockmap_submit"]("region_vhr", {"path": str(path)}, region_id=region_id)
    audit("vhr.upload", f"{region_id}: {path.name}")
    flash("Imagery uploaded; the map layer is being prepared.", "ok")
    return redirect(url_for("regions.region_page", region_id=region_id, job=jid))


@bp.route("/regions/<int:region_id>/rockmap.gpkg")
@requires("viewer")
def region_geopackage(region_id):
    """All vector products as one GeoPackage (rebuilt on request so it is always current)."""
    row = _region_row(region_id)
    region = _region(row)
    p = region.export_geopackage(extra={"field_observations": observations_layer(_db(), region_id)})
    return Response(p.read_bytes(), mimetype="application/geopackage+sqlite3",
                    headers={"Content-Disposition": f"attachment; filename=region{region_id}.gpkg"})


@bp.route("/regions/<int:region_id>/mineral_targets.<fmt>")
@requires("viewer")
def region_mineral_targets_file(region_id, fmt):
    if fmt not in ("csv", "geojson"):
        abort(404)
    p = _region(_region_row(region_id)).folder / "products" / f"mineral_targets.{fmt}"
    if not p.exists():
        abort(404)
    return Response(p.read_bytes(), mimetype="text/csv" if fmt == "csv" else "application/geo+json",
                    headers={"Content-Disposition": f"attachment; filename=region{region_id}_mineral_targets.{fmt}"})


@bp.route("/regions/<int:region_id>/lineaments.geojson")
@requires("viewer")
def region_lineaments_file(region_id):
    p = _region(_region_row(region_id)).folder / "products" / "lineaments.geojson"
    if not p.exists():
        abort(404)
    return Response(p.read_bytes(), mimetype="application/geo+json",
                    headers={"Content-Disposition": f"attachment; filename=region{region_id}_lineaments.geojson"})


@bp.route("/regions/<int:region_id>/gem_targets.<fmt>")
@requires("viewer")
def region_gem_targets_file(region_id, fmt):
    if fmt not in ("csv", "geojson"):
        abort(404)
    p = _region(_region_row(region_id)).folder / "products" / f"gem_targets.{fmt}"
    if not p.exists():
        abort(404)
    return Response(p.read_bytes(), mimetype="text/csv" if fmt == "csv" else "application/geo+json",
                    headers={"Content-Disposition": f"attachment; filename=region{region_id}_gem_targets.{fmt}"})


def _targets(region: Region, name: str = "targets.geojson"):
    p = region.folder / "products" / name
    if not p.exists():
        return jsonify({"type": "FeatureCollection", "features": []})
    return Response(p.read_text(encoding="utf-8"), mimetype="application/json")


def _query(region: Region):
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
    kind = STAGES.get(d.pop("stage", None))
    if not kind:
        return jsonify(error=f"stage must be one of {', '.join(STAGES)}"), 400
    if kind == "region_classify" and not d.get("model_id"):
        return jsonify(error="model_id required"), 400
    job_id = current_app.extensions["rockmap_submit"](kind, d, region_id=region_id)
    return jsonify(job_id=job_id), 202


__all__ = ["bp", "LAYERS", "CLOUD_CLASS", "ALGORITHM_LABELS"]
