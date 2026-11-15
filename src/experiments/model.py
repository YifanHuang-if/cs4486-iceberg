"""Keep baseline_v1's image backbone and add optional metadata fusion."""
import torch
from torch import nn


class ExperimentCNN(nn.Module):
    def __init__(self, dropout=0.25, angle_features=0):
        super().__init__()
        if angle_features not in (0, 1, 2):
            raise ValueError('angle_features must be 0, 1 or 2')
        self.angle_features = angle_features
        # Construction order and parameter names exactly match BaselineCNN when angle_features == 0.
        self.network = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((3, 3)), nn.Flatten(),
            nn.Linear(576, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
        if angle_features:
            self.network[-1] = nn.Identity()
            self.metadata = nn.Sequential(nn.Linear(angle_features, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU())
            self.fusion = nn.Linear(72, 1)

    def forward(self, images, metadata=None):
        features = self.network(images)
        if self.angle_features:
            features = self.fusion(torch.cat((features, self.metadata(metadata[:, :self.angle_features])), dim=1))
        return features.squeeze(1)


def augment_flips(images, generator):
    shape = (len(images), 1, 1, 1)
    horizontal = torch.rand(shape, generator=generator) < 0.5
    vertical = torch.rand(shape, generator=generator) < 0.5
    flipped = torch.where(horizontal, images.flip(-1), images)
    return torch.where(vertical, flipped.flip(-2), flipped)
