import pytest

from rockmap.synthetic import write_scene


@pytest.fixture(scope="session")
def small_scene(tmp_path_factory):
    """A small synthetic study area shared by the tests."""
    out = tmp_path_factory.mktemp("scene")
    return write_scene(out, 160, 160, seed=7, cloud_cover=0.03)


@pytest.fixture(scope="session")
def trained_bundle(small_scene, tmp_path_factory):
    from rockmap.pipeline import train_models
    out = tmp_path_factory.mktemp("model")
    meta = train_models(small_scene["scene"], small_scene["reference"], out, small_scene["dem"],
                        samples_per_class=300, epochs=3, block_size=16)
    return out, meta


@pytest.fixture(autouse=True)
def _isolated_models_dir(tmp_path, monkeypatch):
    """Tests never see (or write to) the real models/trained folder."""
    monkeypatch.setenv("ROCKMAP_MODELS_DIR", str(tmp_path / "trained_models"))
