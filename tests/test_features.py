import numpy as np

from rockmap.features import (INDEX_FEATURES, TERRAIN_FEATURES, Normalizer, build_features,
                              spectral_indices, terrain_features)
from rockmap.preprocessing import (dark_object_subtraction, hot_cloud_mask, landsat_qa_mask, preprocess,
                                   scl_mask, to_reflectance)


def test_to_reflectance_sentinel2_offset():
    dn = np.full((6, 2, 2), 3000, dtype=np.uint16)
    refl = to_reflectance(dn, "sentinel2")
    assert np.allclose(refl, 0.2)  # (3000 - 1000) / 10000


def test_to_reflectance_landsat():
    dn = np.full((6, 1, 1), 10000.0)
    assert np.allclose(to_reflectance(dn, "landsat89"), 10000 * 2.75e-5 - 0.2)


def test_dark_object_subtraction_removes_offset():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 0.3, (6, 50, 50)).astype(np.float32) + 0.05
    out = dark_object_subtraction(img, percentile=0)
    assert np.allclose(out.min(axis=(1, 2)), 0, atol=1e-6)


def test_spectral_indices_ndvi():
    refl = np.zeros((6, 1, 1), np.float32)
    refl[2] = 0.1   # red
    refl[3] = 0.5   # nir
    idx = spectral_indices(refl)
    assert idx.shape[0] == len(INDEX_FEATURES)
    assert np.isclose(idx[0, 0, 0], (0.5 - 0.1) / 0.6, atol=1e-4)


def test_terrain_slope_of_plane():
    yy, xx = np.mgrid[0:20, 0:20].astype(np.float32)
    dem = xx * 20.0  # rises 20 m per 20 m pixel -> 45 degrees
    t = terrain_features(dem, pixel_size=20.0)
    assert t.shape[0] == len(TERRAIN_FEATURES)
    assert np.allclose(t[1, 5:-5, 5:-5] * 90, 45, atol=0.5)


def test_build_features_with_and_without_dem():
    refl = np.random.default_rng(1).uniform(0.05, 0.4, (6, 12, 12)).astype(np.float32)
    f1, n1 = build_features(refl)
    f2, n2 = build_features(refl, np.ones((12, 12), np.float32))
    assert f1.shape[0] == len(n1) == 13
    assert f2.shape[0] == len(n2) == 19
    assert np.isfinite(f2).all()


def test_normalizer_roundtrip():
    x = np.random.default_rng(2).normal(5, 3, (1000, 4)).astype(np.float32)
    n = Normalizer().fit(x)
    z = n.transform_pixels(x)
    assert np.allclose(z.mean(0), 0, atol=1e-4) and np.allclose(z.std(0), 1, atol=1e-3)
    n2 = Normalizer.from_dict(n.to_dict())
    assert np.allclose(n2.transform_pixels(x), z)


def test_cloud_masks():
    refl = np.full((6, 2, 2), 0.2, np.float32)
    refl[:, 0, 0] = [0.5, 0.5, 0.5, 0.5, 0.4, 0.3]            # cloud
    refl[:, 1, 1] = [0.27, 0.31, 0.35, 0.39, 0.46, 0.36]      # bright limestone must survive
    m = hot_cloud_mask(refl)
    assert m[0, 0] and not m[1, 1]
    assert scl_mask(np.array([[4, 9], [3, 11]])).tolist() == [[False, True], [True, False]]
    assert landsat_qa_mask(np.array([21824, 1 << 3, 1 << 5])).tolist() == [False, True, False]


def test_preprocess_requires_six_bands():
    import pytest
    with pytest.raises(ValueError):
        preprocess(np.zeros((4, 3, 3), np.float32))
