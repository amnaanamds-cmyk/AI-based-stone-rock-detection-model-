"""The stand-alone training module: dataset -> train -> versioned model -> prediction / application."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training import config, make_sample_dataset, models as model_cli, train  # noqa: E402
from training.dataset import check_area, find_areas  # noqa: E402

from rockmap import model_store  # noqa: E402

FAST = ["--models", "cnn", "rf", "--epochs", "2", "--samples", "300"]


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path / "processed")
    data = tmp_path / "raw"
    make_sample_dataset.main(["--out", str(data / "area_a"), "--size", "160", "--seed", "3"])
    return data


def test_train_predict_retrain_and_rollback(dataset, tmp_path, capsys):
    models_dir = tmp_path / "models"
    train.main(["--data", str(dataset), "--check"])
    assert "Dataset OK" in capsys.readouterr().out

    v1 = train.main(["--data", str(dataset), "--models-dir", str(models_dir), *FAST])
    assert v1.name == "model_v1" and model_store.current_version(models_dir) == "model_v1"
    meta = json.loads((v1 / "meta.json").read_text())
    for f in ("cnn.pt", "rf.joblib", "evaluation.txt", "evaluation.json", "training_config.json", "cm_cnn.png"):
        assert (v1 / f).exists(), f
    assert meta["sample_data"] is True and meta["app_algorithm"] in ("cnn", "rf")
    assert meta["normalizer"]["mean"] and meta["feature_names"] and meta["uses_dem"] is True
    assert "SYNTHETIC SAMPLE DATA" in (v1 / "evaluation.txt").read_text()
    assert any((tmp_path / "processed").glob("area_a_*.npz"))            # preprocessing cache written

    # prediction module: loads the current model, same preprocessing, writes a map
    from prediction import Predictor
    p = Predictor(models_dir=models_dir)
    area = dataset / "area_a"
    res = p.predict_file(area / "image.tif", tmp_path / "pred", dem=area / "dem.tif")
    with rasterio.open(tmp_path / "pred" / "classified.tif") as src:
        lab = src.read(1)
    assert np.isin(lab, meta["classes"]).mean() > 0.5 and res["algorithm"] == p.algorithm
    with pytest.raises(ValueError):
        p.predict_file(area / "image.tif", tmp_path / "pred2")             # model needs the DEM

    # more data -> retrain -> model_v2 becomes current, v1 is kept (cache reused for area_a)
    make_sample_dataset.main(["--out", str(dataset / "area_b"), "--size", "160", "--seed", "4"])
    v2 = train.main(["--data", str(dataset), "--models-dir", str(models_dir), *FAST])
    assert "cached samples" in capsys.readouterr().out
    assert v2.name == "model_v2" and model_store.versions(models_dir) == ["model_v1", "model_v2"]
    assert model_store.current_version(models_dir) == "model_v2"
    assert len(json.loads((v2 / "meta.json").read_text())["dataset"]) == 2

    # --no-activate keeps the current model; rollback and delete through training/models.py
    v3 = train.main(["--data", str(dataset), "--models-dir", str(models_dir), "--no-activate", *FAST])
    assert model_store.current_version(models_dir) == "model_v2"
    model_cli.main(["use", "model_v1", "--models-dir", str(models_dir)])
    assert model_store.current_version(models_dir) == "model_v1"
    assert Predictor(models_dir=models_dir).version == "model_v1"
    model_cli.main(["delete", v3.name, "--models-dir", str(models_dir)])
    with pytest.raises(SystemExit):
        model_cli.main(["delete", "model_v1", "--models-dir", str(models_dir)])   # current: refused
    model_cli.main(["list", "--models-dir", str(models_dir)])
    assert "* model_v1" in capsys.readouterr().out


def test_vector_labels_with_area_json(dataset, tmp_path):
    """Geological-map polygons + unit mapping instead of a label raster."""
    area = dataset / "area_a"
    with rasterio.open(area / "image.tif") as src:
        b, crs = src.bounds, src.crs
    mx = (b.left + b.right) / 2
    poly = lambda x0, x1: {"type": "Polygon", "coordinates": [[[x0, b.bottom], [x1, b.bottom], [x1, b.top],  # noqa: E731
                                                               [x0, b.top], [x0, b.bottom]]]}
    fc = {"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs.to_string()}},
          "features": [{"type": "Feature", "properties": {"UNIT": "Kohat Limestone"}, "geometry": poly(b.left, mx)},
                       {"type": "Feature", "properties": {"UNIT": "Qa"}, "geometry": poly(mx, b.right)}]}
    (area / "labels.tif").unlink()
    (area / "labels.geojson").write_text(json.dumps(fc))
    (area / "area.json").write_text(json.dumps({"label_field": "UNIT",
                                                "mapping": {"Kohat Limestone": "Limestone", "Qa": 7}}))
    [a] = find_areas(dataset)
    assert a.labels_kind == "vector" and check_area(a) == []
    out = train.main(["--data", str(dataset), "--models-dir", str(tmp_path / "m"), "--models", "rf", "--samples", "200"])
    assert json.loads((out / "meta.json").read_text())["classes"] == [1, 7]


def test_dataset_problems_are_reported(dataset, tmp_path):
    area = dataset / "area_a"
    with rasterio.open(area / "labels.tif") as src:
        prof, lab = src.profile, src.read(1)
    lab[:10, :10] = 42
    with rasterio.open(area / "labels.tif", "w", **prof) as dst:
        dst.write(lab, 1)
    [a] = find_areas(dataset)
    assert any("unknown class ids [42]" in p for p in check_area(a))
    with pytest.raises(SystemExit):
        train.main(["--data", str(dataset), "--models-dir", str(tmp_path / "m")])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="No study area"):
        train.main(["--data", str(empty)])


def test_application_uses_current_model(dataset, tmp_path, monkeypatch):
    models_dir = tmp_path / "models"
    monkeypatch.setenv("ROCKMAP_MODELS_DIR", str(models_dir))
    train.main(["--data", str(dataset), "--models-dir", str(models_dir), "--models", "rf", "--samples", "200"])
    from rockmap.web.app import create_app
    monkeypatch.setenv("ROCKMAP_ADMIN_PASSWORD", "Train-Test-pw-2026!")
    app = create_app(tmp_path / "appdata", sync_jobs=True, workers=0)
    app.config.update(TESTING=True, CSRF_ENABLED=False)
    db = app.extensions["rockmap_db"]
    rows = db.all("models", "source = 'training_module'")
    assert len(rows) == 1 and rows[0]["version"] == "model_v1" and rows[0]["name"].endswith("(current)")
    # retrain while the app is running -> picked up without restart
    train.main(["--data", str(dataset), "--models-dir", str(models_dir), "--models", "rf", "--samples", "200"])
    app.extensions["rockmap_sync_models"](force=True)
    rows = {r["version"]: r for r in db.all("models", "source = 'training_module'")}
    assert rows["model_v2"]["name"].endswith("(current)") and not rows["model_v1"]["name"].endswith("(current)")
    # the region pipeline only falls back to it automatically when it was trained on real data
    from rockmap.web.trained import current_real_model
    assert current_real_model(db) is None

    # pages: model list, model details with the evaluation report and confusion matrix
    c = app.test_client()
    c.post("/login", data={"username": "admin", "password": "Train-Test-pw-2026!"})
    assert "training module" in c.get("/models").get_data(as_text=True)
    page = c.get(f"/models/{rows['model_v2']['id']}").get_data(as_text=True)
    assert "synthetic sample data" in page and "/trained-models/model_v2/cm_rf.png" in page
    assert c.get("/trained-models/model_v2/evaluation.txt").status_code == 200
    assert c.get("/trained-models/model_v2/rf.joblib").status_code == 404          # only reports / images
    r = c.post(f"/models/{rows['model_v2']['id']}/delete", follow_redirects=True)
    assert b"Nothing was deleted" in r.data and (models_dir / "model_v2" / "rf.joblib").exists()
