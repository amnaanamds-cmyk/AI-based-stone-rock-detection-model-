"""Delivery acceptance tests: complete demo build, every page reachable, ops commands."""
import re
from urllib.parse import urljoin, urlparse

import pytest

from rockmap.cli import main
from rockmap.quickstart import build_demo
from rockmap.web.app import create_app

PW = "Acceptance-Test-9"


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    import os
    root = tmp_path_factory.mktemp("delivery")
    old = os.environ.get("ROCKMAP_ADMIN_PASSWORD")
    os.environ["ROCKMAP_ADMIN_PASSWORD"] = PW
    try:
        build_demo(root, size=192, epochs=1)
    finally:
        if old is None:
            os.environ.pop("ROCKMAP_ADMIN_PASSWORD")
        else:
            os.environ["ROCKMAP_ADMIN_PASSWORD"] = old
    return root


def _client(root):
    app = create_app(root, sync_jobs=True, workers=0)
    app.config.update(TESTING=True, CSRF_ENABLED=False)
    c = app.test_client()
    assert c.post("/login", data={"username": "admin", "password": PW}).status_code == 302
    return app, c


def test_demo_is_complete(demo):
    app, c = _client(demo)
    regions = c.get("/api/regions").get_json()
    assert len(regions) == 1
    s = regions[0]["summary"]
    assert s["tiles"] == s["acquired"] == s["classified"] and s["failed"] == 0
    rid = regions[0]["id"]
    folder = demo / regions[0]["folder"]
    for f in ("products/report.pdf", "products/stats.json", "products/analytics.json", "products/lithology.geojson",
              "mosaic/lithology.tif", "mosaic/hazard.tif", "mosaic/alteration.tif", "mosaic/clusters.tif"):
        assert (folder / f).exists(), f
    assert "/" in regions[0]["folder"] and "\\\\" not in regions[0]["folder"]   # portable paths
    assert len(c.get(f"/api/observations?region_id={rid}").get_json()["features"]) == 30


def test_every_page_and_link_works(demo):
    """Crawl the whole dashboard (and the public share page) - no link may return a server error."""
    app, c = _client(demo)
    seen, broken, queue = set(), [], ["/"]
    share = app.extensions["rockmap_db"].all("regions")[0]["share_token"]
    queue.append(f"/share/{share}")
    while queue and len(seen) < 600:
        url = queue.pop()
        if url in seen:
            continue
        seen.add(url)
        r = c.get(url)
        if r.status_code >= 500:
            broken.append((url, r.status_code))
            continue
        if r.status_code in (301, 302):
            loc = urlparse(r.headers["Location"]).path
            if loc and loc not in seen:
                queue.append(loc)
            continue
        if "text/html" not in r.content_type:
            continue
        html = r.get_data(as_text=True)
        for m in re.finditer(r'(?:href|src|data-[a-z-]*api|data-api)="([^"#]+)"', html):
            link = m.group(1).replace("&amp;", "&")
            if link.startswith(("http", "mailto:", "javascript:", "data:")) or "{" in link:
                continue
            path = urljoin(url, link)
            if path.startswith("/logout"):
                continue
            queue.append(path)
    assert not broken, broken
    assert len(seen) > 40            # the crawl really covered the application
    # every map layer renders through the tile server
    rid = app.extensions["rockmap_db"].all("regions")[0]["id"]
    for layer in ("rgb", "falsecolor", "hillshade", "surface", "lithology", "confidence", "alteration", "hazard", "clusters"):
        assert c.get(f"/tiles/{rid}/{layer}/10/700/400.png").status_code == 200


def test_temporary_password_must_be_changed(tmp_path, monkeypatch):
    monkeypatch.delenv("ROCKMAP_ADMIN_PASSWORD", raising=False)
    app = create_app(tmp_path, sync_jobs=True, workers=0)
    app.config.update(TESTING=True, CSRF_ENABLED=False)
    pw = re.search(r"password: (\S+)", (tmp_path / "initial_admin_password.txt").read_text()).group(1)
    c = app.test_client()
    c.post("/login", data={"username": "admin", "password": pw})
    r = c.get("/regions")
    assert r.status_code == 302 and "/profile" in r.headers["Location"]
    assert c.get("/api/regions").status_code == 403
    bad = c.post("/profile", data={"action": "password", "current": pw, "new": "password1", "confirm": "password1"},
                 follow_redirects=True)
    assert b"too common" in bad.data
    c.post("/profile", data={"action": "password", "current": pw, "new": "Strong-Pass-77", "confirm": "Strong-Pass-77"})
    assert c.get("/regions").status_code == 200


def test_doctor_backup_restore(demo, tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["doctor", "--data", str(demo), "--offline", "--port", "5999"])
    out = capsys.readouterr().out
    assert e.value.code == 0 and "All checks passed" in out and "database: 1 users, 1 regions" in out
    zip_path = tmp_path / "b.zip"
    main(["backup", "--data", str(demo), "--out", str(zip_path)])
    target = tmp_path / "restored"
    main(["restore", str(zip_path), "--data", str(target)])
    with pytest.raises(FileExistsError):
        main(["restore", str(zip_path), "--data", str(target)])
    app, c = _client(target)
    region = c.get("/api/regions").get_json()[0]
    assert region["summary"]["classified"] == region["summary"]["tiles"]
    assert c.get(f"/files/{region['folder']}/products/report.pdf").status_code == 200
    assert c.get(f"/tiles/{region['id']}/lithology/10/700/400.png").status_code == 200
