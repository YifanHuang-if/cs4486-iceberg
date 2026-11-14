"""A small CNN trained from scratch, with an optional angle feature branch."""
import torch
from torch import nn


class IcebergCNN(nn.Module):
    def __init__(self, angle_features=0, dropout=0.0):
        super().__init__()
        self.angle_features = angle_features
        blocks = []
        for incoming, outgoing in [(2, 16), (16, 32), (32, 64)]:
            blocks += [nn.Conv2d(incoming, outgoing, 3, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(outgoing), nn.ReLU()]
        self.images = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten())
        self.angle_branch = (nn.Sequential(nn.Linear(angle_features, 16), nn.ReLU(),
                                          nn.Linear(16, 8), nn.ReLU())
                             if angle_features else None)
        self.head = nn.Sequential(nn.Linear(256 + (8 if angle_features else 0), 64),
                                  nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, images, metadata):
        features = self.images(images)
        if self.angle_branch is not None:
            angle = self.angle_branch(metadata[:, :self.angle_features])
            features = torch.cat((features, angle), dim=1)
        return self.head(features).squeeze(1)


def augment_flips(images, generator):
    """Each sample's two radar channels share the same random spatial flips."""
    shape = (len(images), 1, 1, 1)
    horizontal = torch.rand(shape, generator=generator, device=images.device) < .5
    vertical = torch.rand(shape, generator=generator, device=images.device) < .5
    return torch.where(vertical, torch.where(horizontal, images.flip(-1), images).flip(-2),
                       torch.where(horizontal, images.flip(-1), images))
