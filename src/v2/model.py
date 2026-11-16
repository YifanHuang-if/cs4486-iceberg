"""Radar image models and batched geometric augmentation.

The networks are small enough to train many instances on CPU while still having
enough capacity for 75x75 two-channel radar scenes.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

STAT_NAMES = [
    "angle",
    "angle_missing",
    "b0_mean",
    "b1_mean",
    "b0_std",
    "b1_std",
    "b0_p10",
    "b1_p10",
    "b0_p90",
    "b1_p90",
    "b0_max",
    "b1_max",
    "b0_min",
    "b1_min",
    "cross_corr",
]


def image_statistics(images: np.ndarray) -> np.ndarray:
    """Per-image descriptors computed from the two raw bands, shape (N, len(STAT_NAMES))."""
    x = images.astype(np.float64)
    flat0 = x[:, 0].reshape(len(x), -1)
    flat1 = x[:, 1].reshape(len(x), -1)
    corr = np.einsum("ij,ij->i", flat0 - flat0.mean(1, keepdims=True), flat1 - flat1.mean(1, keepdims=True))
    corr = corr / (flat0.std(1) * flat1.std(1) * flat0.shape[1] + 1e-9)
    stats = [
        flat0.mean(1), flat1.mean(1),
        flat0.std(1), flat1.std(1),
        np.percentile(flat0, 10, axis=1), np.percentile(flat1, 10, axis=1),
        np.percentile(flat0, 90, axis=1), np.percentile(flat1, 90, axis=1),
        flat0.max(1), flat1.max(1),
        flat0.min(1), flat1.min(1),
        corr,
    ]
    return np.stack(stats, axis=1).astype(np.float32)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        scale = x.mean(dim=(2, 3), keepdim=True)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(scale))))
        return x * scale


class V1Block(nn.Module):
    """Single 3x3 convolution block used by the compact baseline family."""

    def __init__(self, source: int, target: int, pool: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(source, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.SiLU(),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.body(x))


class RadarV1(nn.Module):
    """Compact five-stage CNN: 93k parameters, three pooling stages, no residuals."""

    def __init__(self, in_channels: int = 2, widths=(16, 32, 64, 96), stats_dim: int = len(STAT_NAMES),
                 dropout: float = 0.3, stem_stride: int = 1):
        super().__init__()
        stages = []
        previous = in_channels
        for index, width in enumerate(widths):
            stages.append(V1Block(previous, width, pool=index < len(widths) - 1))
            previous = width
        if stem_stride == 2:
            stages.insert(0, nn.MaxPool2d(2))
        self.features = nn.Sequential(*stages)
        self.stats = nn.Sequential(nn.Linear(stats_dim, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(widths[-1] * 2 + 16, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, images, stats):
        features = self.features(images)
        pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3)), self.stats(stats)], dim=1)
        return self.head(pooled).squeeze(1)


def build_model(arch: str, **kwargs) -> nn.Module:
    if arch == "v1":
        return RadarV1(**kwargs)
    return RadarNet(**kwargs)


class ConvBlock(nn.Module):
    """Residual squeeze-excite block used by the wider families."""

    def __init__(self, source: int, target: int, pool: bool = True):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(source, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.SiLU(),
            nn.Conv2d(target, target, 3, padding=1, bias=False),
            nn.BatchNorm2d(target),
            nn.SiLU(),
            SqueezeExcite(target),
        )
        self.shortcut = (
            nn.Sequential(nn.Conv2d(source, target, 1, bias=False), nn.BatchNorm2d(target))
            if source != target
            else nn.Identity()
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(self.body(x) + self.shortcut(x))


class RadarNet(nn.Module):
    def __init__(self, in_channels: int = 2, widths=(24, 48, 96, 160), stats_dim: int = len(STAT_NAMES),
                 dropout: float = 0.3, stem_stride: int = 1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, widths[0], 3, stride=stem_stride, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.SiLU(),
        )
        blocks = []
        previous = widths[0]
        for index, width in enumerate(widths):
            blocks.append(ConvBlock(previous, width, pool=index < len(widths) - 1))
            previous = width
        self.blocks = nn.Sequential(*blocks)
        self.stats = nn.Sequential(nn.Linear(stats_dim, 32), nn.SiLU(), nn.Linear(32, 16), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(widths[-1] * 2 + 16, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def forward(self, images, stats):
        features = self.blocks(self.stem(images))
        pooled = torch.cat([features.mean(dim=(2, 3)), features.amax(dim=(2, 3)), self.stats(stats)], dim=1)
        return self.head(pooled).squeeze(1)


def view(images: torch.Tensor, code: int) -> torch.Tensor:
    """One of the eight dihedral symmetries."""
    if code & 1:
        images = images.transpose(-1, -2)
    if code & 2:
        images = images.flip(-1)
    if code & 4:
        images = images.flip(-2)
    return images


def augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Dihedral flips plus a small random similarity transform, applied per sample."""
    shape = (len(images), 1, 1, 1)
    for code, action in ((1, "transpose"), (2, "horizontal"), (4, "vertical")):
        mask = (torch.rand(shape, generator=generator) < 0.5).to(images.device)
        changed = images.transpose(-1, -2) if action == "transpose" else images.flip(
            -1 if action == "horizontal" else -2
        )
        images = torch.where(mask, changed, images)
    values = torch.rand((len(images), 5), generator=generator).to(images.device)
    radians = (values[:, 0] - 0.5) * (math.pi / 4)
    scale = 0.9 + values[:, 1] * 0.2
    matrix = torch.zeros((len(images), 2, 3), device=images.device)
    matrix[:, 0, 0] = scale * radians.cos()
    matrix[:, 0, 1] = -scale * radians.sin()
    matrix[:, 1, 0] = scale * radians.sin()
    matrix[:, 1, 1] = scale * radians.cos()
    matrix[:, :, 2] = (values[:, 2:4] - 0.5) * 0.2
    grid = F.affine_grid(matrix, images.shape, align_corners=False)
    images = F.grid_sample(images, grid, padding_mode="reflection", align_corners=False)
    noise = torch.randn(images.shape, generator=generator).to(images.device) * 0.05
    return images + noise


def predict(model: nn.Module, images: torch.Tensor, stats: torch.Tensor, device, tta: bool = True, batch: int = 256) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = images[start:start + batch].to(device)
            s = stats[start:start + batch].to(device)
            codes = range(8) if tta else [0]
            views = [torch.sigmoid(model(view(x, code), s)).cpu().numpy().astype(np.float64) for code in codes]
            outputs.append(np.mean(views, axis=0))
    return np.concatenate(outputs)
