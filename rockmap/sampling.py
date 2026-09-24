"""Training data preparation from a reference geological map.

Pixels are split into train / validation / test sets by *spatial blocks* rather than at
random: neighbouring pixels are strongly correlated, so a random split would leak
information and give over-optimistic accuracies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CLASS_IDS, DEFAULT_BLOCK_SIZE, DEFAULT_SAMPLES_PER_CLASS


@dataclass
class SampleSet:
    rows: np.ndarray
    cols: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def block_split(shape: tuple[int, int], block: int = DEFAULT_BLOCK_SIZE, fractions=(0.6, 0.15, 0.25),
                seed: int = 0) -> np.ndarray:
    """Assign every pixel to split 0 (train), 1 (val) or 2 (test) by square blocks."""
    h, w = shape
    nby, nbx = -(-h // block), -(-w // block)
    rng = np.random.default_rng(seed)
    ids = rng.permutation(nby * nbx)
    n_train = int(round(fractions[0] * len(ids)))
    n_val = int(round(fractions[1] * len(ids)))
    split_of_block = np.full(len(ids), 2, dtype=np.uint8)
    split_of_block[ids[:n_train]] = 0
    split_of_block[ids[n_train:n_train + n_val]] = 1
    grid = split_of_block.reshape(nby, nbx)
    return np.repeat(np.repeat(grid, block, 0), block, 1)[:h, :w]


def sample_pixels(labels: np.ndarray, valid: np.ndarray, split_map: np.ndarray, split: int,
                  per_class: int = DEFAULT_SAMPLES_PER_CLASS, margin: int = 0, seed: int = 0,
                  erode_boundaries: bool = True) -> SampleSet:
    """Stratified random sampling of labelled pixels belonging to ``split``.

    ``margin`` keeps samples away from the image border (needed for CNN patches).
    ``erode_boundaries`` skips pixels on lithological contacts, where reference maps
    are least reliable.
    """
    rng = np.random.default_rng(seed + 17 * split)
    h, w = labels.shape
    ok = valid & (split_map == split) & (labels > 0)
    if margin:
        ok[:margin] = ok[-margin:] = False
        ok[:, :margin] = ok[:, -margin:] = False
    if erode_boundaries:
        pad = np.pad(labels, 1, mode="edge")
        same = np.ones_like(ok)
        for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
            same &= pad[dy:dy + h, dx:dx + w] == labels
        ok &= same
    rows, cols, labs = [], [], []
    for c in CLASS_IDS:
        r, cc = np.nonzero(ok & (labels == c))
        if len(r) == 0:
            continue
        take = rng.choice(len(r), size=min(per_class, len(r)), replace=False)
        rows.append(r[take]); cols.append(cc[take]); labs.append(np.full(len(take), c))
    if not rows:
        return SampleSet(np.empty(0, int), np.empty(0, int), np.empty(0, int))
    rows, cols, labs = map(np.concatenate, (rows, cols, labs))
    order = rng.permutation(len(labs))
    return SampleSet(rows[order], cols[order], labs[order].astype(np.int64))


def pixel_vectors(features: np.ndarray, samples: SampleSet) -> np.ndarray:
    """(n_samples, n_features) matrix for classical classifiers."""
    return features[:, samples.rows, samples.cols].T.copy()


def extract_patches(features: np.ndarray, samples: SampleSet, patch: int) -> np.ndarray:
    """(n_samples, n_features, patch, patch) array centred on each sample pixel."""
    r = patch // 2
    padded = np.pad(features, ((0, 0), (r, r), (r, r)), mode="reflect")
    offs = np.arange(-r, r + 1)
    rr = samples.rows[:, None, None] + r + offs[None, :, None]
    cc = samples.cols[:, None, None] + r + offs[None, None, :]
    return padded[:, rr, cc].transpose(1, 0, 2, 3).copy()
