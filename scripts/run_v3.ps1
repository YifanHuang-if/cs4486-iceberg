param(
    [string]$Python = ".venv\Scripts\python.exe",
    [string]$RawDirectory = "data\raw",
    [string]$Device = "cpu",
    [int]$Threads = 4,
    [switch]$RunStudent
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
function Invoke-CheckedPython {
    param([string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $Arguments" }
}
Invoke-CheckedPython -Arguments @("-m","src.v3.prepare","--raw",$RawDirectory)
Invoke-CheckedPython -Arguments @("-m","unittest","src.v3.test_isolation","-q")
Invoke-CheckedPython -Arguments @("-m","src.v3.train","--tag","v3_mild_s11","--epochs","40","--threads","$Threads","--device",$Device)
Invoke-CheckedPython -Arguments @("-m","src.v3.train","--tag","v3_global_s29","--seed","29","--normalisation","global","--augmentation","affine","--epochs","40","--threads","$Threads","--device",$Device)
Invoke-CheckedPython -Arguments @("-m","src.v3.evaluate","--tags","v3_mild_s11","v3_global_s29","--name","v3_supervised_ensemble")
if ($RunStudent) {
    Invoke-CheckedPython -Arguments @("-m","src.v3.train","--tag","v3_student_s47","--seed","47","--teacher","v3_mild_s11","--initialize-teacher","--lr","0.0003","--epochs","15","--threads","$Threads","--device",$Device)
    Invoke-CheckedPython -Arguments @("-m","src.v3.evaluate","--tags","v3_student_s47","--name","v3_student")
    Invoke-CheckedPython -Arguments @("-m","src.v3.evaluate","--tags","v3_mild_s11","v3_global_s29","v3_student_s47","--name","v3_all_ensemble")
}
