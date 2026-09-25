import json

import numpy as np
import rasterio

from rockmap.cli import main
from rockmap.pipeline import ModelBundle, classify_scene, load_scene
from rockmap.synthetic import generate_scene


def test_synthetic_scene_contents():
    s = generate_scene(96, 96, seed=3)
    assert s["reflectance"].shape == (6, 96, 96)
    assert set(np.unique(s["reference"])) <= set(range(1, 8))
    assert len(np.unique(s["truth"])) >= 5
    assert s["dem"].std() > 10


def test_scene_files(small_scene):
    with rasterio.open(small_scene["scene"]) as src:
        assert src.count == 6 and src.crs.to_epsg() == 32642


def test_training_bundle(trained_bundle):
    folder, meta = trained_bundle
    assert set(meta["algorithms"]) == {"cnn", "rf", "svm"}
    for algo in ("cnn", "rf", "svm"):
        assert meta["algorithms"][algo]["test_metrics"]["overall_accuracy"] > 0.5
        assert (folder / f"cm_{algo}.png").exists()
    assert json.loads((folder / "meta.json").read_text())["uses_dem"] is True


def test_bundle_predict_all_algorithms(trained_bundle, small_scene):
    bundle = ModelBundle(trained_bundle[0])
    scene = load_scene(small_scene["scene"], small_scene["dem"])
    for algo in bundle.algorithms:
        labels, conf = bundle.predict(scene, algo)
        assert labels.shape == (160, 160)
        assert set(np.unique(labels)) <= set(range(1, 8)) | {250, 251, 252, 253, 255}
        assert 0 <= conf.min() and conf.max() <= 1.0001


def test_classify_scene_with_window_and_validation(trained_bundle, small_scene, tmp_path):
    res = classify_scene(trained_bundle[0], small_scene["scene"], tmp_path, "rf", small_scene["dem"],
                         reference_path=small_scene["reference"], window=[10, 20, 100, 80])
    assert (res["width"], res["height"]) == (100, 80)
    assert res["validation"]["overall_accuracy"] > 0.6
    for name in ("classified.tif", "classified.png", "confidence.tif", "rgb.png", "agreement.png",
                 "map_figure.png", "result.json"):
        assert (tmp_path / name).exists(), name
    with rasterio.open(tmp_path / "classified.tif") as src:
        assert src.width == 100 and src.transform.c == 620000 + 10 * 20


def test_cli_classify_and_evaluate(trained_bundle, small_scene, tmp_path, capsys):
    main(["classify", "--model", str(trained_bundle[0]), "--scene", str(small_scene["scene"]),
          "--dem", str(small_scene["dem"]), "--algorithm", "cnn", "--out", str(tmp_path)])
    main(["evaluate", "--pred", str(tmp_path / "classified.tif"), "--ref", str(small_scene["reference"]),
          "--json", str(tmp_path / "m.json")])
    out = capsys.readouterr().out
    assert "Overall accuracy" in out
    assert json.loads((tmp_path / "m.json").read_text())["n_samples"] > 1000


def test_model_requires_dem(trained_bundle, small_scene):
    import pytest
    bundle = ModelBundle(trained_bundle[0])
    with pytest.raises(ValueError, match="DEM"):
        bundle.predict(load_scene(small_scene["scene"]), "rf")


def test_blockwise_classification_matches_single_block(trained_bundle, small_scene, tmp_path):
    """Processing in small blocks (with halo) must give the same map as one big block."""
    a = classify_scene(trained_bundle[0], small_scene["scene"], tmp_path / "a", "cnn", small_scene["dem"], block=1024)
    b = classify_scene(trained_bundle[0], small_scene["scene"], tmp_path / "b", "cnn", small_scene["dem"], block=48)
    with rasterio.open(tmp_path / "a" / "classified.tif") as s1, rasterio.open(tmp_path / "b" / "classified.tif") as s2:
        la, lb = s1.read(1), s2.read(1)
    assert (la == lb).mean() > 0.995
    assert a["area_stats"] == b["area_stats"] or abs(a["mean_confidence"] - b["mean_confidence"]) < 0.01
