import numpy as np

from rockmap.evaluation import compare_maps, confusion_matrix, evaluate
from rockmap.mapping import area_statistics, colorize, majority_filter
from rockmap.sampling import SampleSet, block_split, extract_patches, sample_pixels


def test_block_split_fractions_and_blocks():
    s = block_split((128, 128), block=16, seed=0)
    assert set(np.unique(s)) == {0, 1, 2}
    # every block is homogeneous
    blocks = s.reshape(8, 16, 8, 16).transpose(0, 2, 1, 3).reshape(64, -1)
    assert all(len(np.unique(b)) == 1 for b in blocks)
    assert 0.5 < (s == 0).mean() < 0.7


def test_sample_pixels_stratified_and_in_split():
    labels = np.zeros((64, 64), np.uint8)
    labels[:, :32] = 1
    labels[:, 32:] = 5
    split = block_split(labels.shape, 16, seed=1)
    s = sample_pixels(labels, np.ones_like(labels, bool), split, 0, per_class=50, margin=4)
    assert set(np.unique(s.labels)) == {1, 5}
    assert (np.bincount(s.labels)[[1, 5]] == 50).all()
    assert (split[s.rows, s.cols] == 0).all()
    assert s.rows.min() >= 4 and s.rows.max() < 60


def test_extract_patches_centre():
    feats = np.arange(2 * 10 * 10, dtype=np.float32).reshape(2, 10, 10)
    s = SampleSet(np.array([5, 0]), np.array([5, 0]), np.array([1, 1]))
    p = extract_patches(feats, s, 3)
    assert p.shape == (2, 2, 3, 3)
    assert p[0, 0, 1, 1] == feats[0, 5, 5]
    assert p[0, 1, 0, 0] == feats[1, 4, 4]


def test_metrics_perfect_and_confusion():
    y = np.array([1, 1, 2, 2, 3])
    m = evaluate(y, y)
    assert m["overall_accuracy"] == 1.0 and np.isclose(m["kappa"], 1.0)
    cm = confusion_matrix(np.array([1, 2]), np.array([2, 2]))
    assert cm[0, 1] == 1 and cm[1, 1] == 1


def test_kappa_known_value():
    # classic 2-class example: OA 0.7, pe 0.5 -> kappa 0.4
    t = np.array([1] * 50 + [2] * 50)
    p = np.array([1] * 35 + [2] * 15 + [2] * 35 + [1] * 15)
    m = evaluate(t, p)
    assert np.isclose(m["overall_accuracy"], 0.7) and np.isclose(m["kappa"], 0.4)


def test_compare_maps_ignores_masked():
    ref = np.array([[1, 2], [0, 3]], np.uint8)
    pred = np.array([[1, 255], [4, 3]], np.uint8)
    m = compare_maps(pred, ref)
    assert m["n_samples"] == 2 and m["overall_accuracy"] == 1.0


def test_majority_filter_removes_speckle():
    lab = np.full((9, 9), 2, np.uint8)
    lab[4, 4] = 5
    assert (majority_filter(lab, 3) == 2).all()


def test_colorize_and_stats():
    lab = np.array([[1, 1], [7, 255]], np.uint8)
    rgba = colorize(lab)
    assert rgba.shape == (2, 2, 4) and rgba[1, 1, 3] == 0
    stats = {s["id"]: s for s in area_statistics(lab, 0.0004)}
    assert np.isclose(stats[1]["percent"], 200 / 3) and stats[255]["pixels"] == 1
