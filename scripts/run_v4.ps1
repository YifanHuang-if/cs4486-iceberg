# Run the V4 reliability diagnostic using existing V3 predictions.
# This script does not retrain neural networks or upload competition data.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python virtual environment missing: $python"
}

$required = @(
    "work/cache/train.npz",
    "work/cache/test.npz",
    "data/splits/folds.csv"
)
$tags = @("v3_mild_s11", "v3_global_s29", "v3_shape_mlp")
foreach ($tag in $tags) {
    foreach ($fold in 0..4) {
        $required += "outputs/v3/$tag/fold${fold}_predictions.npz"
        $required += "outputs/v3/$tag/fold${fold}_metrics.json"
    }
}
$missing = @($required | Where-Object { -not (Test-Path (Join-Path $root $_)) })
if ($missing.Count -gt 0) {
    Write-Host "The V4 experiment needs saved V3 OOF artifacts:" -ForegroundColor Yellow
    $missing | ForEach-Object { Write-Host "  $_" }
    throw "Missing artifacts. Do not rerun training until you locate the existing V3 predictions."
}

& $python -m unittest src.v4.test_reliability src.v4.test_groups -v
if ($LASTEXITCODE -ne 0) { throw "V4 regression tests failed." }

& $python -m src.v4.evaluate_reliability --tags v3_mild_s11 v3_global_s29 v3_shape_mlp --name v4_reliability_fixed
if ($LASTEXITCODE -ne 0) { throw "V4 experiment failed." }

Write-Host "Saved: results/v4/v4_reliability_fixed.json" -ForegroundColor Green
