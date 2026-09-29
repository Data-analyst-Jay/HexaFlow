param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    $machine = & $Python -c "import platform; print(platform.machine().lower())"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not run the selected Python interpreter."
    }

    if ($machine.Trim() -notin @("arm64", "aarch64")) {
        throw "Build must run with native Windows ARM64 Python. Found: $machine"
    }

    # Fail before packaging if the required ARM64 runtime imports are missing.
    & $Python -c "import geniex; import qai_appbuilder; import onnxruntime; import PyInstaller"

    if ($LASTEXITCODE -ne 0) {
        throw "ARM64 packaging preflight failed."
    }

    Write-Host "ARM64 QAIRT/GenieX/PyInstaller preflight passed."

    & $Python -m PyInstaller `
        --noconfirm `
        --clean `
        --distpath dist `
        --workpath build `
        hexaflow.spec

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed."
    }

    $output = Join-Path $projectRoot "dist\WisprFlowClone.exe"
    if (-not (Test-Path -LiteralPath $output)) {
        throw "Build completed without creating $output"
    }

    Write-Host "Created $output"
}
finally {
    Pop-Location
}