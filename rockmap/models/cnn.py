"""Convolutional Neural Network for lithological classification.

``LithoCNN`` is a *fully convolutional* patch classifier: four unpadded 3x3 convolutions
give a 9x9 receptive field, so the network is trained on 9x9 patches (predicting the
centre pixel) and at inference time slides over a whole scene in a single pass, producing
a per-pixel class probability map. This combines the spatial/texture context a CNN learns
with the efficiency of dense prediction.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn


class LithoCNN(nn.Module):
    def __init__(self, in_channels: int, n_classes: int, width: int = 64, dropout: float = 0.3):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

        self.stem = nn.Sequential(  # 1x1 "spectral" mixing before spatial convolutions
            nn.Conv2d(in_channels, width, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True))
        self.features = nn.Sequential(
            block(width, width), block(width, width), block(width, 2 * width), block(2 * width, 2 * width))
        self.head = nn.Sequential(nn.Dropout2d(dropout), nn.Conv2d(2 * width, 2 * width, 1),
                                  nn.ReLU(inplace=True), nn.Conv2d(2 * width, n_classes, 1))

    receptive_field = 9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(self.stem(x)))


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Random flips and 90-degree rotations (geology has no preferred orientation)."""
    if torch.rand(1).item() < 0.5:
        x = x.flip(-1)
    if torch.rand(1).item() < 0.5:
        x = x.flip(-2)
    k = int(torch.randint(0, 4, (1,)).item())
    return torch.rot90(x, k, dims=(-2, -1)) if k else x


class CNNClassifier:
    def __init__(self, in_channels: int, classes: list[int], patch: int = 9, width: int = 64,
                 epochs: int = 30, batch_size: int = 256, lr: float = 2e-3, seed: int = 0,
                 device: Optional[str] = None):
        if patch != LithoCNN.receptive_field:
            raise ValueError(f"patch size must equal the CNN receptive field ({LithoCNN.receptive_field})")
        torch.manual_seed(seed)
        self.classes = np.asarray(classes)
        self.patch = patch
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = LithoCNN(in_channels, len(classes), width).to(self.device)
        self.hparams = dict(in_channels=in_channels, classes=list(map(int, classes)), patch=patch, width=width)
        self.history: list[dict] = []
        self.train_seconds = 0.0

    def _to_index(self, y: np.ndarray) -> np.ndarray:
        lut = {c: i for i, c in enumerate(self.classes)}
        return np.array([lut[int(v)] for v in y], dtype=np.int64)

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            progress: Optional[Callable[[int, int, dict], None]] = None) -> "CNNClassifier":
        """x: (N, C, patch, patch) normalised patches, y: class ids."""
        xt = torch.from_numpy(x).float()
        yt = torch.from_numpy(self._to_index(y))
        counts = np.bincount(yt.numpy(), minlength=len(self.classes)).astype(np.float32)
        weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / len(counts), device=self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        steps = self.epochs * max(1, -(-len(yt) // self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
        best_state, best_acc = None, -1.0
        t0 = time.time()
        for epoch in range(self.epochs):
            self.net.train()
            perm = torch.randperm(len(yt))
            total, correct, loss_sum = 0, 0, 0.0
            for i in range(0, len(perm), self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = _augment(xt[idx]).to(self.device)
                yb = yt[idx].to(self.device)
                logits = self.net(xb).flatten(1)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
                loss_sum += loss.item() * len(idx)
                correct += (logits.argmax(1) == yb).sum().item()
                total += len(idx)
            rec = {"epoch": epoch + 1, "loss": loss_sum / total, "train_acc": correct / total}
            if x_val is not None and len(x_val):
                pred = self.predict_patches(x_val).argmax(1)
                rec["val_acc"] = float((self.classes[pred] == y_val).mean())
                if rec["val_acc"] > best_acc:
                    best_acc = rec["val_acc"]
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            self.history.append(rec)
            if progress:
                progress(epoch + 1, self.epochs, rec)
        if best_state is not None:  # keep the epoch with the best validation accuracy
            self.net.load_state_dict(best_state)
        self.train_seconds = time.time() - t0
        return self

    @torch.no_grad()
    def predict_patches(self, x: np.ndarray, batch: int = 2048) -> np.ndarray:
        self.net.eval()
        out = []
        for i in range(0, len(x), batch):
            xb = torch.from_numpy(x[i:i + batch]).float().to(self.device)
            out.append(torch.softmax(self.net(xb).flatten(1), 1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), np.float32)

    @torch.no_grad()
    def predict_image(self, features: np.ndarray, tile: int = 512) -> np.ndarray:
        """Normalised (F, H, W) features -> (n_classes, H, W) probabilities (tiled, dense)."""
        self.net.eval()
        r = self.patch // 2
        f, h, w = features.shape
        padded = np.pad(features, ((0, 0), (r, r), (r, r)), mode="reflect")
        probs = np.zeros((len(self.classes), h, w), dtype=np.float32)
        for y0 in range(0, h, tile):
            for x0 in range(0, w, tile):
                y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
                chunk = padded[:, y0:y1 + 2 * r, x0:x1 + 2 * r]
                xb = torch.from_numpy(np.ascontiguousarray(chunk))[None].float().to(self.device)
                probs[:, y0:y1, x0:x1] = torch.softmax(self.net(xb), 1)[0].cpu().numpy()
        return probs

    def save(self, path) -> None:
        torch.save({"hparams": self.hparams, "state_dict": self.net.state_dict(),
                    "history": self.history, "train_seconds": self.train_seconds}, path)

    @classmethod
    def load(cls, path, device: Optional[str] = None) -> "CNNClassifier":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        hp = ckpt["hparams"]
        obj = cls(hp["in_channels"], hp["classes"], hp["patch"], hp["width"], device=device)
        obj.net.load_state_dict(ckpt["state_dict"])
        obj.net.eval()
        obj.history = ckpt.get("history", [])
        obj.train_seconds = ckpt.get("train_seconds", 0.0)
        return obj
