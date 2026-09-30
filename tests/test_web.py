import io
import json
import re

import numpy as np
import pytest

from rockmap.web.app import create_app
from rockmap.web.auth import create_user

ADMIN_PW = "admin-password-123"


@pytest.fixture()
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCKMAP_ADMIN_PASSWORD", ADMIN_PW)
    app = create_app(tmp_path, sync_jobs=True)
    app.config.update(TESTING=True, CSRF_ENABLED=False)
    return app


def login(client, user="admin", pw=ADMIN_PW):
    return client.post("/login", data={"username": user, "password": pw})


@pytest.fixture()
def client(app):
    c = app.test_client()
    assert login(c).status_code == 302
    return c


def test_login_required_and_api_401(app):
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 302 and "/login" in r.headers["Location"]
    assert c.get("/api/regions").status_code == 401
    assert c.get("/healthz").get_json()["status"] == "ok"
    r = login(c, "admin", "wrong-password")
    assert b"Invalid username or password" in c.get("/login").data or r.status_code == 200


def test_login_lockout(app):
    c = app.test_client()
    for _ in range(5):
        login(c, "admin", "nope")
    r = login(c)  # correct password now refused
    assert r.status_code == 200 and b"Too many failed attempts" in r.data


def test_roles(app):
    db = app.extensions["rockmap_db"]
    create_user(db, "viewer1", "viewer-pass-1", "viewer")
    c = app.test_client()
    login(c, "viewer1", "viewer-pass-1")
    assert c.get("/").status_code == 200
    assert c.get("/scenes/upload").status_code == 403
    assert c.post("/scenes/demo").status_code == 403
    assert c.get("/admin/users").status_code == 403


def test_csrf_enforced(app):
    app.config["CSRF_ENABLED"] = True
    c = app.test_client()
    page = c.get("/login").get_data(as_text=True)
    token = re.search(r'name="csrf_token" value="([^"]+)"', page).group(1)
    assert c.post("/login", data={"username": "admin", "password": ADMIN_PW}).status_code == 400
    r = c.post("/login", data={"username": "admin", "password": ADMIN_PW, "csrf_token": token})
    assert r.status_code == 302


def test_api_token(client, app):
    html = client.post("/profile", data={"action": "token"}).get_data(as_text=True)
    token = re.search(r"(rmk_[A-Za-z0-9_\-]+)", html).group(1)
    anon = app.test_client()
    r = anon.get("/api/me", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.get_json()["username"] == "admin"
    assert anon.get("/api/me", headers={"Authorization": "Bearer rmk_wrong"}).status_code == 401


def test_admin_user_management(client, app):
    r = client.post("/admin/users", data={"action": "create", "username": "ana", "password": "analyst-pw-1",
                                          "role": "analyst"}, follow_redirects=True)
    assert b"User created" in r.data
    db = app.extensions["rockmap_db"]
    uid = db.one("users", "username = ?", ("ana",))["id"]
    client.post("/admin/users", data={"action": "deactivate", "user_id": uid})
    c2 = app.test_client()
    assert login(c2, "ana", "analyst-pw-1").status_code == 200  # disabled -> login refused
    assert client.get("/admin/audit").status_code == 200


def test_scene_flow(client, small_scene):
    assert client.get("/").status_code == 200
    data = {"name": "Test area", "sensor": "reflectance"}
    for k in ("scene", "dem", "reference"):
        data[k] = (open(small_scene[k], "rb"), f"{k}.tif")
    r = client.post("/scenes/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 302
    scene_url = r.headers["Location"]
    page = client.get(scene_url).get_data(as_text=True)
    assert "Test area" in page and "Train models" in page
    sid = int(scene_url.rstrip("/").split("/")[-1])

    r = client.post(f"/scenes/{sid}/train", data={"algorithms": ["rf", "cnn"], "epochs": "2",
                                                  "samples": "200", "use_dem": "on"})
    job_page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Benchmark" in job_page, job_page[:2000]
    models = client.get("/api/models").get_json()
    assert len(models) == 1 and set(models[0]["summary"]) == {"rf", "cnn"}

    r = client.post(f"/scenes/{sid}/classify", data={"model_id": models[0]["id"], "algorithm": "rf",
                                                     "smoothing": "3", "window": "0,0,100,90"})
    job_url = r.headers["Location"]
    html = client.get(job_url).get_data(as_text=True)
    assert "Validation against reference geological map" in html
    job_id = int(job_url.split("/")[-1])
    assert client.get(f"/api/jobs/{job_id}").get_json()["status"] == "done"
    png = re.search(r'src="([^"]+classified\.png)"', html).group(1)
    assert client.get(png).status_code == 200
    assert client.get(f"/models/{models[0]['id']}").status_code == 200
    assert client.get("/jobs").status_code == 200


def test_upload_rejects_bad_scene(client):
    r = client.post("/scenes/upload", data={"scene": (io.BytesIO(b"not a tiff"), "x.txt")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert "Upload failed" in r.get_data(as_text=True)


def test_file_route_is_restricted(client):
    assert client.get("/files/rockmap.db").status_code == 404
    assert client.get("/files/scenes/../rockmap.db").status_code == 404
    assert client.get("/files/secret_key").status_code == 404


def test_cancel_queued_job(client, app):
    db = app.extensions["rockmap_db"]
    jid = db.insert("jobs", kind="region_acquire", status="queued", params={})
    client.post(f"/jobs/{jid}/cancel")
    assert db.get("jobs", jid)["status"] == "cancelled"


def test_region_pipeline_in_dashboard(client, app, small_scene):
    """Admin creates a region from server-side files, draws training areas, runs the full pipeline."""
    from rasterio.features import shapes
    from rasterio.warp import transform_geom
    from rockmap.io import read_raster
    _, info = read_raster(small_scene["scene"], bands=[1])
    w, s, e, n = info.wgs84_bounds()
    r = client.post("/regions/new", data={
        "aoi_mode": "draw", "aoi_geojson": json.dumps({"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}),
        "name": "Test valley", "resolution": "20", "tile_size": "128", "source": "local",
        "local_scenes": str(small_scene["scene"]), "local_sensor": "reflectance", "local_dem": str(small_scene["dem"])})
    assert r.status_code == 302, r.data
    rid = int(r.headers["Location"].split("/")[-1])
    assert client.get(f"/regions/{rid}").status_code == 200

    # training areas from the synthetic reference map (one polygon per class)
    ref, rinfo = read_raster(small_scene["reference"])
    lab = np.nan_to_num(ref[0]).astype(np.uint8)
    seen = {}
    for geom, val in shapes(lab, mask=lab > 0, transform=rinfo.transform):
        v = int(val)
        if len(geom["coordinates"][0]) > 8 and seen.get(v, 0) < 2:
            g = transform_geom(rinfo.crs, "EPSG:4326", geom)
            r = client.post(f"/api/regions/{rid}/annotations", json={"class_id": v, "geometry": g})
            assert r.status_code == 201, r.get_json()
            seen[v] = seen.get(v, 0) + 1
    assert len(client.get(f"/api/regions/{rid}/annotations").get_json()["features"]) >= 5
    bad = client.post(f"/api/regions/{rid}/annotations", json={"class_id": 250, "geometry": g})
    assert bad.status_code == 400

    r = client.post(f"/regions/{rid}/run", data={"stage": "pipeline", "algorithms": ["rf"], "epochs": "1",
                                                 "samples": "300", "smoothing": "3", "geojson": "on"})
    assert r.status_code == 302
    jobs = client.get("/api/jobs").get_json()
    assert jobs[0]["status"] == "done", jobs[0]
    summary = client.get(f"/api/regions/{rid}").get_json()["summary"]
    assert summary["classified"] == summary["tiles"] > 1
    stats = client.get(f"/api/regions/{rid}/stats").get_json()
    assert sum(s["area_km2"] for s in stats["region"]) > 5
    assert client.get(f"/regions/{rid}/stats.csv").status_code == 200
    q = client.get(f"/api/regions/{rid}/query?lat={(s + n) / 2}&lon={(w + e) / 2}").get_json()
    assert q["inside"] and "class_name" in q
    tiles = client.get(f"/api/regions/{rid}/tiles").get_json()
    assert len(tiles["features"]) == summary["tiles"]
    # an XYZ tile over the region centre
    import math
    z = 12
    lat, lon = (s + n) / 2, (w + e) / 2
    x = int((lon + 180) / 360 * 2 ** z)
    y = int((1 - math.log(math.tan(math.radians(lat)) + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * 2 ** z)
    t = client.get(f"/tiles/{rid}/lithology/{z}/{x}/{y}.png")
    assert t.status_code == 200 and t.data[:4] == b"\x89PNG" and len(t.data) > 500
    page = client.get(f"/regions/{rid}").get_data(as_text=True)
    assert "PDF report" in page and "Polygons (GeoJSON)" in page
    row = app.extensions["rockmap_db"].get("regions", rid)
    pdf = client.get(f"/files/{row['folder']}/products/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"


def test_region_presets_create(client):
    r = client.post("/regions/new", data={"aoi_mode": "preset", "preset": "gilgit-baltistan"})
    rid = int(r.headers["Location"].split("/")[-1])
    s = client.get(f"/api/regions/{rid}").get_json()
    assert s["config"]["epsg"] == 32643
    assert 150 < s["summary"]["tiles"] < 260       # ~73,000 km2 in 20.48 km tiles


def _local_region(client, small_scene, tile="128"):
    from rockmap.io import read_raster
    _, info = read_raster(small_scene["scene"], bands=[1])
    w, s, e, n = info.wgs84_bounds()
    r = client.post("/regions/new", data={
        "aoi_mode": "draw", "aoi_geojson": json.dumps({"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}),
        "name": "Analytics valley", "resolution": "20", "tile_size": tile, "source": "local",
        "local_scenes": str(small_scene["scene"]), "local_sensor": "reflectance", "local_dem": str(small_scene["dem"])})
    return int(r.headers["Location"].split("/")[-1]), (w, s, e, n)


def test_analytics_units_share_and_styles(client, app, small_scene):
    rid, (w, s, e, n) = _local_region(client, small_scene)
    client.post(f"/regions/{rid}/run", data={"stage": "acquire"})
    client.post(f"/regions/{rid}/run", data={"stage": "analyze", "n_clusters": "5"})
    page = client.get(f"/regions/{rid}").get_data(as_text=True)
    assert "Spectral units" in page and "Landslide / rockfall susceptibility" in page and "Key findings" in page
    client.post(f"/regions/{rid}/units", data={"unit_1": "4", "unit_2": "7", "unit_3": "1"})
    assert client.get(f"/api/regions/{rid}").get_json()["summary"]["classified"] > 0
    assert client.get(f"/regions/{rid}/targets.csv").status_code == 200
    assert client.get(f"/api/regions/{rid}/targets").get_json()["type"] == "FeatureCollection"
    qml = client.get(f"/regions/{rid}/style/lithology.qml").get_data(as_text=True)
    assert "paletteEntry" in qml and "Granite" in qml
    assert client.get(f"/regions/{rid}/style/hazard.qml").status_code == 200
    # public share link: works without login, stops working after revoke
    client.post(f"/regions/{rid}/share", data={"action": "create"})
    token = app.extensions["rockmap_db"].get("regions", rid)["share_token"]
    anon = app.test_client()
    assert anon.get(f"/share/{token}").status_code == 200
    assert anon.get(f"/share/{token}/api/query?lat={(s + n) / 2}&lon={(w + e) / 2}").get_json()["inside"]
    assert anon.get(f"/share/{token}/tiles/hazard/12/0/0.png").status_code == 200
    assert anon.get(f"/api/regions/{rid}").status_code == 401
    client.post(f"/regions/{rid}/share", data={"action": "revoke"})
    assert anon.get(f"/share/{token}").status_code == 404


def test_field_observations(client, app, small_scene):
    import base64
    import io as _io
    from PIL import Image
    rid, (w, s, e, n) = _local_region(client, small_scene, "160")
    buf = _io.BytesIO()
    Image.new("RGB", (40, 30), (120, 80, 40)).save(buf, "JPEG")
    photo = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    r = client.post("/api/observations", json={"lat": (s + n) / 2, "lon": (w + e) / 2, "class_id": 4, "certainty": 3,
                                               "note": "granite outcrop", "photo_data": photo})
    assert r.status_code == 201 and r.get_json()["region_id"] == rid      # region found from location
    assert client.post("/api/observations", json={"lat": 95, "lon": 0, "class_id": 4}).status_code == 400
    gj = client.get(f"/api/observations?region_id={rid}").get_json()
    assert len(gj["features"]) == 1 and gj["features"][0]["properties"]["photo"]
    assert client.get(gj["features"][0]["properties"]["photo"]).status_code == 200
    assert client.get("/field/").status_code == 200 and client.get("/field/sw.js").status_code == 200
    # viewers cannot add, other analysts cannot delete
    create_user(app.extensions["rockmap_db"], "view2", "viewer-pass-2", "viewer")
    v = app.test_client()
    login(v, "view2", "viewer-pass-2")
    assert v.post("/api/observations", json={"lat": 35, "lon": 74, "class_id": 1}).status_code == 403
    oid = gj["features"][0]["properties"]["id"]
    assert client.delete(f"/api/observations/{oid}").get_json()["ok"]


def test_gem_workflow_in_dashboard(client, app, small_scene):
    rid, (w, s, e, n) = _local_region(client, small_scene)
    client.post(f"/regions/{rid}/run", data={"stage": "acquire"})
    csv_data = f"lat,lon,gem,name\n{(s + n) / 2},{(w + e) / 2},ruby,Test locality\n".encode()
    r = client.post(f"/regions/{rid}/gem-occurrences", data={"file": (io.BytesIO(csv_data), "known.csv"),
                                                             "mode": "replace"},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"1 known gem localities saved" in r.data
    # a field find with a gem is also a known locality
    assert client.post("/api/observations", json={"lat": (s + n) / 2, "lon": (w + e) / 2 + 0.001, "class_id": 1,
                                                   "gem": "spinel"}).status_code == 201
    assert client.post("/api/observations", json={"lat": (s + n) / 2, "lon": (w + e) / 2, "class_id": 1,
                                                   "gem": "kryptonite"}).status_code == 400
    occ = client.get(f"/api/regions/{rid}/gem-occurrences").get_json()
    assert len(occ["features"]) == 2
    client.post(f"/regions/{rid}/run", data={"stage": "gems"})
    page = client.get(f"/regions/{rid}").get_data(as_text=True)
    assert "Gemstone prospectivity" in page and "Marble-hosted ruby" in page
    assert client.get(f"/api/regions/{rid}/gem-targets").get_json()["type"] == "FeatureCollection"
    assert client.get(f"/regions/{rid}/gem_targets.csv").status_code == 200
    assert client.get(f"/tiles/{rid}/gems/12/0/0.png").status_code == 200
    assert client.get(f"/tiles/{rid}/gem_pegmatite/12/0/0.png").status_code == 200
    field = client.get("/field/").get_data(as_text=True)
    assert "aquamarine" in field.lower()


def test_mineral_workflow_in_dashboard(client, app, small_scene, tmp_path):
    rid, (w, s, e, n) = _local_region(client, small_scene)
    client.post(f"/regions/{rid}/run", data={"stage": "acquire"})
    csv_data = f"lat,lon,commodity,name\n{(s + n) / 2},{(w + e) / 2},copper,Test prospect\n".encode()
    r = client.post(f"/regions/{rid}/mineral-occurrences", data={"file": (io.BytesIO(csv_data), "occ.csv"),
                                                                 "mode": "replace"},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"1 known mineral occurrences saved" in r.data
    assert client.post("/api/observations", json={"lat": (s + n) / 2, "lon": (w + e) / 2, "class_id": 1,
                                                   "commodity": "antimony"}).status_code == 201
    assert client.post("/api/observations", json={"lat": (s + n) / 2, "lon": (w + e) / 2, "class_id": 1,
                                                   "commodity": "unobtainium"}).status_code == 400
    assert len(client.get(f"/api/regions/{rid}/mineral-occurrences").get_json()["features"]) == 2
    client.post(f"/regions/{rid}/run", data={"stage": "minerals"})
    page = client.get(f"/regions/{rid}").get_data(as_text=True)
    assert "Mineral prospectivity" in page and "Structural lineaments" in page
    assert client.get(f"/api/regions/{rid}/mineral-targets").get_json()["type"] == "FeatureCollection"
    assert client.get(f"/api/regions/{rid}/lineaments").get_json()["type"] == "FeatureCollection"
    assert client.get(f"/regions/{rid}/mineral_targets.csv").status_code == 200
    assert client.get(f"/regions/{rid}/lineaments.geojson").status_code == 200
    for layer in ("minerals", "min_vein", "lineaments"):
        assert client.get(f"/tiles/{rid}/{layer}/12/0/0.png").status_code == 200

    # ASTER emissivity upload -> Quartz Index -> vein model uses it
    import numpy as np
    import rasterio
    from rockmap.io import read_raster
    _, info = read_raster(small_scene["scene"])
    emis = np.stack([np.full((info.height, info.width), v, np.float32) for v in (0.8, 0.9, 0.8)])
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=info.width, height=info.height, count=3, dtype="float32",
                      crs=info.crs, transform=info.transform) as dst:
            dst.write(emis)
        buf.write(mem.read())
    buf.seek(0)
    r = client.post(f"/regions/{rid}/aster", data={"files": (buf, "ast05.tif")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"ASTER file(s) uploaded" in r.data
    assert "ASTER in use" in client.get(f"/regions/{rid}").get_data(as_text=True)
    field = client.get("/field/").get_data(as_text=True)
    assert "antimony" in field.lower()


def test_hyperspectral_vhr_and_geopackage(client, app, small_scene, tmp_path):
    import numpy as np
    import rasterio
    from rockmap.gpkg import layer_names
    from rockmap.io import read_raster
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("hs_helpers", Path(__file__).with_name("test_hyperspectral.py"))
    hs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hs)
    WL, _cube_like = hs.WL, hs._cube_like
    rid, _ = _local_region(client, small_scene)
    client.post(f"/regions/{rid}/run", data={"stage": "acquire"})
    _, info = read_raster(small_scene["scene"])
    cube = _cube_like(info, tmp_path / "enmap.tif", lambda r: "alunite" if r % 2 else "chlorite")
    hdr = "wavelength = {" + ", ".join(f"{w:.1f}" for w in WL) + "}"
    r = client.post(f"/regions/{rid}/hyperspectral",
                    data={"scenes": (io.BytesIO(cube.read_bytes()), "enmap.tif"),
                          "wavelengths": (io.BytesIO(hdr.encode()), "enmap.hdr")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"hyperspectral scene(s) uploaded" in r.data
    page = client.get(f"/regions/{rid}").get_data(as_text=True)
    assert "Hyperspectral alteration minerals" in page and "Alunite" in page
    assert client.get(f"/tiles/{rid}/hyper/12/0/0.png").status_code == 200

    # very-high-resolution imagery: 4-band (B, G, R, NIR) 2 m image over the region
    vhr = tmp_path / "vhr.tif"
    t = info.transform
    with rasterio.open(vhr, "w", driver="GTiff", width=200, height=200, count=4, dtype="uint16", crs=info.crs,
                       transform=rasterio.transform.from_origin(t.c, t.f, 2.0, 2.0)) as dst:
        dst.write(np.random.default_rng(0).integers(200, 3000, (4, 200, 200)).astype(np.uint16))
    r = client.post(f"/regions/{rid}/vhr", data={"file": (io.BytesIO(vhr.read_bytes()), "wv3.tif")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"Imagery uploaded" in r.data
    assert "current: wv3.tif" in client.get(f"/regions/{rid}").get_data(as_text=True)
    assert client.get(f"/tiles/{rid}/vhr/15/0/0.png").status_code == 200

    r = client.get(f"/regions/{rid}/rockmap.gpkg")
    assert r.status_code == 200 and r.data[:15] == b"SQLite format 3"
    p = tmp_path / "out.gpkg"
    p.write_bytes(r.data)
    assert "area_of_interest" in layer_names(p)
