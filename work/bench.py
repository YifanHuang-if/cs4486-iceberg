import time
import torch
from src.v2.model import RadarNet, augment

torch.set_num_threads(16)


def bench(arch_kwargs, batch, steps=8, aug=True):
    model = RadarNet(**arch_kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch, 2, 75, 75)
    s = torch.randn(batch, 15)
    y = torch.randint(0, 2, (batch,)).float()
    criterion = torch.nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(0)
    model.train()
    t0 = time.perf_counter()
    for _ in range(steps):
        xb = augment(x, g) if aug else x
        opt.zero_grad(set_to_none=True)
        loss = criterion(model(xb, s), y)
        loss.backward()
        opt.step()
    train_time = (time.perf_counter() - t0) / steps
    model.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(steps):
            model(x, s)
        eval_time = (time.perf_counter() - t0) / steps
    params = sum(p.numel() for p in model.parameters())
    print(f"{arch_kwargs.get('widths')} stride={arch_kwargs.get('stem_stride',1)} batch={batch} aug={aug}: "
          f"train {train_time*1000:.0f}ms/step  eval {eval_time*1000:.0f}ms/step  params={params/1e3:.0f}k")


if __name__ == "__main__":
    bench(dict(widths=(16, 32, 64, 128)), 64, aug=False)
    bench(dict(widths=(16, 32, 64, 128)), 64, aug=True)
    bench(dict(widths=(32, 64, 128, 192), stem_stride=2), 64, aug=True)
    bench(dict(widths=(32, 64, 128, 192), stem_stride=2), 128, aug=True)
    bench(dict(widths=(24, 48, 96, 160)), 128, aug=True)
