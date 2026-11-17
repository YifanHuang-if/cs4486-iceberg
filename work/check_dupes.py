import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
train = np.load(ROOT / "work/cache/train.npz", allow_pickle=False)
test = np.load(ROOT / "work/cache/test.npz", allow_pickle=False)


def views(x):
    out = []
    for code in range(8):
        v = x.copy()
        if code & 1:
            v = v.transpose(0, 1, 3, 2)
        if code & 2:
            v = v[:, :, :, ::-1]
        if code & 4:
            v = v[:, :, ::-1, :]
        out.append(v)
    return out


def thumb(images):
    x = images.reshape(len(images), 2, 15, 5, 15, 5).mean(axis=(3, 5))
    flat = x.reshape(len(x), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


train_t = thumb(train["images"])
test_t = thumb(test["images"])

best = np.full(len(test_t), -np.inf)
best_i = np.zeros(len(test_t), dtype=int)
best_code = np.zeros(len(test_t), dtype=int)
for code in range(8):
    v = test_t.reshape(len(test_t), 2, 15, 15).copy()
    if code & 1:
        v = v.transpose(0, 1, 3, 2)
    if code & 2:
        v = v[:, :, :, ::-1]
    if code & 4:
        v = v[:, :, ::-1, :]
    flat = v.reshape(len(test_t), -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)
    score = flat @ train_t.T
    idx = score.argmax(axis=1)
    val = score[np.arange(len(score)), idx]
    better = val > best
    best = np.where(better, val, best)
    best_i = np.where(better, idx, best_i)
    best_code = np.where(better, code, best_code)

rows = np.flatnonzero(best > 0.99)
print("test rows with similarity > 0.99:", len(rows))
for row in rows[:12]:
    a = test["images"][row]
    b = train["images"][best_i[row]]
    code = int(best_code[row])
    if code & 1:
        a = a.transpose(0, 2, 1)
    if code & 2:
        a = a[:, :, ::-1]
    if code & 4:
        a = a[:, ::-1, :]
    print(f"test {test['ids'][row]} train {train['ids'][best_i[row]]} label={int(train['labels'][best_i[row]])} "
          f"sim={best[row]:.6f} maxabs={np.abs(a-b).max():.6f} equal={np.array_equal(a,b)}")
