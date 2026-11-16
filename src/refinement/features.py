"""Per-image radar statistics; no labels or cross-sample statistics are used."""
import hashlib
from pathlib import Path
import numpy as np
from scipy import ndimage
from src.experiments.data import ROOT


def radar_features(images, angles):
    rows = []
    coordinates = np.indices((75, 75)).astype(np.float64)
    for image, angle in zip(images, angles):
        row = {'angle': float(angle) if np.isfinite(angle) else -1., 'angle_missing': float(not np.isfinite(angle))}
        planes = [image[0], image[1], image[0]-image[1]]
        for channel, plane in enumerate(planes):
            prefix = f'c{channel}_'; a = plane.astype(np.float64)
            mean, std = a.mean(), max(a.std(), 1e-6)
            q = np.quantile(a, [.01, .1, .25, .5, .75, .9, .95, .99, .999])
            row.update({prefix+'mean': mean, prefix+'std': std, prefix+'min': a.min(), prefix+'max': a.max(),
                        prefix+'skew': np.mean(((a-mean)/std)**3), prefix+'kurtosis': np.mean(((a-mean)/std)**4),
                        prefix+'center_mean': a[25:50,25:50].mean(),
                        prefix+'edge_mean': np.concatenate([a[:8].ravel(),a[-8:].ravel(),a[:, :8].ravel(),a[:, -8:].ravel()]).mean(),
                        prefix+'roughness': (np.abs(np.diff(a, axis=0)).mean()+np.abs(np.diff(a,axis=1)).mean())/2})
            row.update({prefix+f'q{index}': value for index,value in enumerate(q)})
            smoothed = ndimage.gaussian_filter(a, sigma=1)
            row[prefix+'smooth_std'] = smoothed.std()
            row[prefix+'smooth_max'] = smoothed.max()
            for k, multiple in enumerate([1.,2.,3.]):
                mask = smoothed > np.median(smoothed)+multiple*max(smoothed.std(),1e-6)
                components, count = ndimage.label(mask)
                sizes = np.bincount(components.ravel())[1:]
                row[prefix+f'area{k}'] = mask.mean()
                row[prefix+f'components{k}'] = count
                row[prefix+f'largest{k}'] = sizes.max()/a.size if len(sizes) else 0.
            weight = np.maximum(smoothed - np.quantile(smoothed,.9), 0)
            mass = weight.sum()
            if mass > 1e-12:
                centre = (coordinates*weight).sum(axis=(1,2))/mass
                offset = coordinates-centre[:,None,None]
                covariance = (offset.reshape(2,-1)*weight.ravel())@offset.reshape(2,-1).T/mass
                eig = np.linalg.eigvalsh(covariance)
                row[prefix+'extent'] = eig.sum()
                row[prefix+'elongation'] = (eig[1]+1)/(eig[0]+1)
                row[prefix+'offcentre'] = np.linalg.norm(centre-37)
            else:
                row.update({prefix+'extent': 0.,prefix+'elongation':1.,prefix+'offcentre':0.})
        row['cross_correlation'] = float(np.mean((image[0]-image[0].mean())*(image[1]-image[1].mean())) / max(float(image[0].std()*image[1].std()),1e-6))
        rows.append(row)
    names = list(rows[0])
    result = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float32)
    if not np.isfinite(result).all(): raise ValueError('Invalid radar statistics')
    return result,names


def training_features(data):
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path = ROOT/f'outputs/refinement_cache/features_{data["sha256"]}_{source}.npz'
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path,allow_pickle=False) as archive:
            assert np.array_equal(archive['ids'],data['ids'])
            return archive['features'],archive['names'].tolist()
    features,names=radar_features(data['images'],data['angles'])
    np.savez(path,features=features,names=np.asarray(names),ids=data['ids'])
    return features,names
