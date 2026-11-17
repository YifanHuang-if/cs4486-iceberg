"""How sensitive is the required near-row improvement to the exact-row assumption?

The round-7 decomposition assumed the cross-validated exact-row loss (0.0223)
also holds on the test exact rows. That is an assumption, not a measurement, so
this sweeps it and reports the range of the implied test near-row loss and of
the improvement still required to reach 0.083.
"""
from __future__ import annotations

import json

import numpy as np

from src.v3.train import ROOT

MEASURED_PRIVATE = 0.08944
TARGET = 0.083
SHARE_EXACT = 0.5695
SHARE_NEAR = 0.4171


def main() -> int:
    report = {"measured_private": MEASURED_PRIVATE, "target": TARGET,
              "share_exact": SHARE_EXACT, "share_near": SHARE_NEAR, "rows": []}
    print(f"{'assumed exact loss':>18s} {'implied near loss':>18s} "
          f"{'required near':>14s} {'improvement needed':>19s}")
    for assumed_exact in (0.010, 0.015, 0.0223, 0.030, 0.040):
        implied_near = (MEASURED_PRIVATE - SHARE_EXACT * assumed_exact) / SHARE_NEAR
        required_near = (TARGET - SHARE_EXACT * assumed_exact) / SHARE_NEAR
        improvement = (implied_near - required_near) / implied_near
        entry = {"assumed_exact": assumed_exact,
                 "implied_near": round(implied_near, 4),
                 "required_near": round(required_near, 4),
                 "improvement": round(improvement, 4)}
        report["rows"].append(entry)
        print(f"{assumed_exact:18.4f} {implied_near:18.4f} {required_near:14.4f} "
              f"{improvement:18.1%}")

    implied = [row["implied_near"] for row in report["rows"]]
    improvement = [row["improvement"] for row in report["rows"]]
    report["implied_near_range"] = [min(implied), max(implied)]
    report["improvement_range"] = [min(improvement), max(improvement)]
    print(f"\nimplied test near loss spans {min(implied):.3f} to {max(implied):.3f}")
    print(f"required relative improvement spans {min(improvement):.1%} to {max(improvement):.1%}")
    (ROOT / "work/leak/lb_sensitivity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
