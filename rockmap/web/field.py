"""Field data collection (mobile, offline-capable) and field validation of maps."""
from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

from flask import Blueprint, Response, current_app, g, jsonify, render_template, request, url_for
from PIL import Image

from ..config import ALL_CLASSES, CLASS_IDS, ROCK_CLASSES
from ..region import Region
from .auth import audit, has_role, requires
from .db import Database

bp = Blueprint("field", __name__)
MAX_PHOTO_PX = 1600


def _db() -> Database:
    return current_app.extensions["rockmap_db"]


def _root() -> Path:
    return current_app.config["DATA_DIR"]


def region_for_point(lon: float, lat: float):
    """First region whose tile grid contains the point (or None)."""
    for r in _db().all("regions"):
        b = r.get("bounds")
        if b and not (b[0] <= lon <= b[2] and b[1] <= lat <= b[3]):
            continue
        try:
            if Region(_root() / r["folder"]).query(lon, lat).get("inside"):
                return r["id"]
        except (OSError, ValueError, KeyError):
            continue
    return None


def observation_squares(db: Database, region_id: int, half_m: float = 20.0) -> list:
    """Confident field observations as small squares (training polygons)."""
    out = []
    for o in db.all("observations", "region_id = ? AND certainty >= 2", (region_id,)):
        dlat = half_m / 111_320.0
        dlon = half_m / (111_320.0 * max(0.2, math.cos(math.radians(o["lat"]))))
        ring = [[o["lon"] - dlon, o["lat"] - dlat], [o["lon"] + dlon, o["lat"] - dlat],
                [o["lon"] + dlon, o["lat"] + dlat], [o["lon"] - dlon, o["lat"] + dlat], [o["lon"] - dlon, o["lat"] - dlat]]
        out.append(({"type": "Polygon", "coordinates": [ring]}, o["class_id"]))
    return out


def _save_photo(obs_id: int, data: bytes) -> str:
    img = Image.open(io.BytesIO(data))
    img = img.convert("RGB")
    img.thumbnail((MAX_PHOTO_PX, MAX_PHOTO_PX))
    folder = _root() / "observations"
    folder.mkdir(exist_ok=True)
    name = f"{obs_id}.jpg"
    img.save(folder / name, "JPEG", quality=85)
    return f"observations/{name}"


def observations_geojson(rows: list[dict]) -> dict:
    return {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": {"type": "Point", "coordinates": [o["lon"], o["lat"]]},
         "properties": {"id": o["id"], "class_id": o["class_id"], "class": ALL_CLASSES[o["class_id"]].name,
                        "color": ALL_CLASSES[o["class_id"]].color, "certainty": o["certainty"], "note": o["note"],
                        "author": o["author"], "observed_at": o["observed_at"] or o["created"],
                        "photo": url_for("files", relpath=o["photo"]) if o["photo"] else None}}
        for o in rows]}


@bp.route("/field/")
@requires("viewer")
def field_page():
    return render_template("field.html", classes=ROCK_CLASSES, regions=_db().all("regions"),
                           recent=_db().all("observations", "author = ?", (g.user["username"],), limit=15))


@bp.route("/field/sw.js")
def service_worker():
    """Service worker: caches the field page and its assets so it opens without a network."""
    assets = [url_for("field.field_page"), url_for("static", filename="style.css"),
              url_for("static", filename="field.js"), url_for("static", filename="vendor/leaflet/leaflet.js"),
              url_for("static", filename="vendor/leaflet/leaflet.css")]
    js = f"""const CACHE = "rockmap-field-v2";
const ASSETS = {json.dumps(assets)};
self.addEventListener("install", e => e.waitUntil(caches.open(CACHE).then(c => c.addAll(ASSETS)).then(() => self.skipWaiting())));
self.addEventListener("activate", e => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", e => {{
  if (e.request.method !== "GET") return;
  const url = new URL(e.request.url);
  if (!ASSETS.includes(url.pathname)) return;
  e.respondWith(fetch(e.request).then(r => {{ const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); return r; }})
    .catch(() => caches.match(e.request)));
}});"""
    resp = Response(js, mimetype="application/javascript")
    resp.headers["Service-Worker-Allowed"] = "/field/"
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@bp.route("/field/manifest.json")
def manifest():
    return jsonify({"name": "RockMap Field", "short_name": "RockMap", "start_url": url_for("field.field_page"),
                    "scope": "/field/", "display": "standalone", "background_color": "#f6f4ef",
                    "theme_color": "#b5562b", "icons": []})


@bp.route("/api/observations", methods=["GET", "POST"])
@requires("viewer")
def api_observations():
    db = _db()
    if request.method == "GET":
        rid = request.args.get("region_id", type=int)
        rows = db.all("observations", "region_id = ?", (rid,)) if rid else db.all("observations", limit=2000)
        return jsonify(observations_geojson(rows))
    if not has_role("analyst"):
        return jsonify(error="'analyst' role required"), 403
    d = request.form.to_dict() if request.form else (request.get_json(silent=True) or {})
    try:
        lat, lon = float(d["lat"]), float(d["lon"])
        cid = int(d["class_id"])
        certainty = int(d.get("certainty") or 2)
    except (KeyError, TypeError, ValueError):
        return jsonify(error="lat, lon and class_id are required"), 400
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or cid not in CLASS_IDS or certainty not in (1, 2, 3):
        return jsonify(error="invalid coordinates, class or certainty"), 400
    region_id = int(d["region_id"]) if str(d.get("region_id") or "").isdigit() else region_for_point(lon, lat)
    acc = d.get("accuracy_m")
    oid = db.insert("observations", region_id=region_id, lat=lat, lon=lon,
                    accuracy_m=float(acc) if acc not in (None, "", "null") else None, class_id=cid,
                    certainty=certainty, note=(d.get("note") or "")[:1000],
                    observed_at=(d.get("observed_at") or "")[:32] or None, author=g.user["username"])
    photo = request.files.get("photo")
    try:
        if photo and photo.filename:
            db.update("observations", oid, photo=_save_photo(oid, photo.read()))
        elif str(d.get("photo_data", "")).startswith("data:image"):
            db.update("observations", oid, photo=_save_photo(oid, base64.b64decode(d["photo_data"].split(",", 1)[1])))
    except (OSError, ValueError):
        pass  # keep the observation even if the photo is unreadable
    audit("observation.create", f"#{oid} class {cid} at {lat:.5f},{lon:.5f}")
    return jsonify(id=oid, region_id=region_id), 201


@bp.route("/api/observations/<int:oid>", methods=["DELETE"])
@requires("analyst")
def api_observation_delete(oid):
    db = _db()
    o = db.get("observations", oid)
    if not o:
        return jsonify(error="not found"), 404
    if o["author"] != g.user["username"] and not has_role("admin"):
        return jsonify(error="only the author or an admin can delete this observation"), 403
    if o["photo"]:
        (_root() / o["photo"]).unlink(missing_ok=True)
    db.delete("observations", oid)
    audit("observation.delete", f"#{oid}")
    return jsonify(ok=True)
