"""Radar CNN and fold-local preprocessing; independently implemented."""
from dataclasses import dataclass,asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

METADATA = ['angle','angle_missing','c0_mean','c1_mean','c0_std','c1_std','c0_q3','c1_q3','c0_q7','c1_q7','c2_mean','c2_std']


@dataclass
class Preparation:
    image_mean: list
    image_std: list
    metadata_mean: list
    metadata_std: list
    angle_median: float
    normalization: str

    @classmethod
    def fit(cls, images, features, normalization):
        observed=features[:,0][features[:,1]==0]
        median=float(np.median(observed)) if len(observed) else 0.
        features=features.astype(np.float64).copy();features[features[:,1]==1,0]=median
        return cls(images.mean((0,2,3),dtype=np.float64).tolist(),
                   images.std((0,2,3),dtype=np.float64).clip(1e-6).tolist(),
                   features.mean(0).tolist(),features.std(0).clip(1e-6).tolist(),median,normalization)

    def transform(self, images, features):
        x=images.astype(np.float32)
        if self.normalization=='local':
            mean=x.mean((2,3),keepdims=True,dtype=np.float64).astype(np.float32)
            std=x.std((2,3),keepdims=True,dtype=np.float64).astype(np.float32).clip(1e-6)
        else:
            mean=np.asarray(self.image_mean,dtype=np.float32)[None,:,None,None]
            std=np.asarray(self.image_std,dtype=np.float32)[None,:,None,None]
        x=(x-mean)/std
        f=features.astype(np.float64).copy();f[f[:,1]==1,0]=self.angle_median
        f=(f-np.asarray(self.metadata_mean))/np.asarray(self.metadata_std)
        return np.ascontiguousarray(x,dtype=np.float32),np.ascontiguousarray(f,dtype=np.float32)

    def to_dict(self):return asdict(self)


class RadarCNN(nn.Module):
    def __init__(self, metadata_features=len(METADATA)):
        super().__init__()
        layers=[]
        channels=[2,16,32,64,96]
        for i,(source,target) in enumerate(zip(channels[:-1],channels[1:])):
            layers += [nn.Conv2d(source,target,3,padding=1,bias=False),nn.BatchNorm2d(target),nn.SiLU()]
            if i<3:layers.append(nn.MaxPool2d(2))
        self.features=nn.Sequential(*layers)
        self.metadata=nn.Sequential(nn.Linear(metadata_features,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(208,64),nn.SiLU(),nn.Dropout(.3),nn.Linear(64,1))
    def forward(self,images,metadata):
        features=self.features(images)
        combined=torch.cat([features.mean((-1,-2)),features.amax((-1,-2)),self.metadata(metadata)],dim=1)
        return self.head(combined).squeeze(1)


def view(images,code):
    if code&1:images=images.transpose(-1,-2)
    if code&2:images=images.flip(-1)
    if code&4:images=images.flip(-2)
    return images


def augment(images,generator):
    # Draws are on CPU, with a private generator independent of dropout and sample order.
    shape=(len(images),1,1,1)
    for action in ['transpose','horizontal','vertical']:
        mask=(torch.rand(shape,generator=generator)<.5).to(images.device)
        changed=images.transpose(-1,-2) if action=='transpose' else images.flip(-1 if action=='horizontal' else -2)
        images=torch.where(mask,changed,images)
    values=torch.rand((len(images),4),generator=generator).to(images.device)
    radians=(values[:,0]-.5)*(.34906585)
    scale=.92+values[:,1]*.16
    matrix=torch.zeros((len(images),2,3),device=images.device)
    matrix[:,0,0]=scale*radians.cos();matrix[:,0,1]=-scale*radians.sin()
    matrix[:,1,0]=scale*radians.sin();matrix[:,1,1]=scale*radians.cos()
    matrix[:,:,2]=(values[:,2:]-.5)*.16
    grid=F.affine_grid(matrix,images.shape,align_corners=False)
    return F.grid_sample(images,grid,padding_mode='reflection',align_corners=False)


def predict(model,x,metadata,device,tta=False):
    model.eval();results=[]
    with torch.no_grad():
        for start in range(0,len(x),128):
            images=x[start:start+128].to(device);features=metadata[start:start+128].to(device)
            probabilities=[model(view(images,code),features).sigmoid().cpu().numpy().astype(np.float64) for code in (range(8) if tta else [0])]
            results.append(np.mean(probabilities,axis=0))
    return np.concatenate(results)
