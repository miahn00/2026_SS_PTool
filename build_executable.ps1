$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "프로젝트 가상환경 Python을 찾을 수 없습니다: $PythonExe"
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "SS_PTool_V0.0.1" `
    --paths (Join-Path $ProjectRoot "Source") `
    --collect-all matplotlib `
    --collect-all pyqtgraph `
    --hidden-import "scipy._external.array_api_compat.numpy.fft" `
    --hidden-import "scipy._external.array_api_compat.numpy.linalg" `
    (Join-Path $ProjectRoot "Source\main.py")

if ($LASTEXITCODE -ne 0) {
    throw "실행파일 빌드에 실패했습니다. Exit code: $LASTEXITCODE"
}

Write-Host "빌드 완료: $ProjectRoot\dist\SS_PTool_V0.0.1\SS_PTool_V0.0.1.exe"
