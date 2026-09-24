import io
import re

import numpy as np
import pytest

from rockmap.web.app import create_app


@pytest.fixture()
def client(tmp_path):
    app = create_app(tmp_path, sync_jobs=True)
    app.config["TESTING"] = True
    return app.test_client()


def test_full_dashboard_flow(client, small_scene):
    assert client.get("/").status_code == 200
    # upload the synthetic scene with DEM and reference map
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
    assert "Legend" in html
    job_id = int(job_url.split("/")[-1])
    assert client.get(f"/api/jobs/{job_id}").get_json()["status"] == "done"
    png = re.search(r'src="([^"]+classified\.png)"', html).group(1)
    assert client.get(png).status_code == 200
    assert client.get(f"/models/{models[0]['id']}").status_code == 200
    assert client.get("/models").status_code == 200


def test_upload_rejects_bad_scene(client, tmp_path):
    r = client.post("/scenes/upload", data={"scene": (io.BytesIO(b"not a tiff"), "x.txt")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert "Upload failed" in r.get_data(as_text=True)


def test_file_route_is_restricted(client):
    assert client.get("/files/rockmap.db").status_code == 404
    assert client.get("/files/scenes/../rockmap.db").status_code == 404


def test_demo_scene_and_about(client):
    r = client.post("/scenes/demo", data={"size": "128", "seed": "1"})
    assert r.status_code == 302
    assert "Reference map" in client.get(r.headers["Location"]).get_data(as_text=True)
    assert client.get("/about").status_code == 200
    assert len(client.get("/api/classes").get_json()) == 7
