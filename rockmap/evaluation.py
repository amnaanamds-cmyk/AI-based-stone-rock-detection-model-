"""Accuracy assessment against reference geological maps."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .config import CLASS_IDS, class_name


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes=CLASS_IDS) -> np.ndarray:
    """Rows = reference class, columns = predicted class."""
    n = len(classes)
    lut = np.full(256, -1, np.int64)
    lut[np.asarray(classes, dtype=np.int64)] = np.arange(n)
    t = lut[np.clip(np.ravel(y_true).astype(np.int64), 0, 255)]
    p = lut[np.clip(np.ravel(y_pred).astype(np.int64), 0, 255)]
    ok = (t >= 0) & (p >= 0)
    return np.bincount(t[ok] * n + p[ok], minlength=n * n).reshape(n, n)


def metrics_from_confusion(cm: np.ndarray, classes=CLASS_IDS) -> dict:
    total = cm.sum()
    diag = np.diag(cm).astype(float)
    oa = diag.sum() / total if total else 0.0
    pe = (cm.sum(0) * cm.sum(1)).sum() / total ** 2 if total else 0.0
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    per_class = []
    f1s = []
    for i, c in enumerate(classes):
        support = int(cm[i].sum())
        precision = diag[i] / cm[:, i].sum() if cm[:, i].sum() else 0.0   # user's accuracy
        recall = diag[i] / support if support else 0.0                     # producer's accuracy
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        if support:
            f1s.append(f1)
        per_class.append({"id": int(c), "name": class_name(c), "precision": float(precision),
                          "recall": float(recall), "f1": float(f1), "support": support})
    return {
        "overall_accuracy": float(oa),
        "kappa": float(kappa),
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "n_samples": int(total),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "classes": [int(c) for c in classes],
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, classes=CLASS_IDS) -> dict:
    return metrics_from_confusion(confusion_matrix(y_true, y_pred, classes), classes)


def compare_maps(predicted: np.ndarray, reference: np.ndarray, mask: Optional[np.ndarray] = None,
                 classes=CLASS_IDS) -> dict:
    """Pixel-wise validation of a classified map against a reference geological map."""
    ok = (reference > 0) & np.isin(predicted, classes)
    if mask is not None:
        ok &= mask
    return evaluate(reference[ok], predicted[ok], classes)


def format_report(m: dict, title: str = "") -> str:
    lines = []
    if title:
        lines += [title, "=" * len(title)]
    lines.append(f"Overall accuracy : {m['overall_accuracy'] * 100:6.2f} %")
    lines.append(f"Cohen's kappa    : {m['kappa']:.4f}")
    lines.append(f"Macro F1         : {m['macro_f1']:.4f}")
    lines.append(f"Samples          : {m['n_samples']}")
    lines.append(f"{'Class':34s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'Support':>8s}")
    for c in m["per_class"]:
        lines.append(f"{c['name']:34s} {c['precision']:6.3f} {c['recall']:6.3f} {c['f1']:6.3f} {c['support']:8d}")
    return "\n".join(lines)
