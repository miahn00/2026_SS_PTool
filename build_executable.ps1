$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$BridgeProject = Join-Path $ProjectRoot "CameraBridge\CameraBridge.csproj"
$BridgeOutput = Join-Path $ProjectRoot "CameraBridge\bin\Release\net8.0-windows"
$CamDirectory = Join-Path $ProjectRoot "reference_dll\SDK_Package\cam File"

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "프로젝트 가상환경 Python을 찾을 수 없습니다: $PythonExe"
}

$DotnetExe = if ($env:SS_PTOOL_DOTNET) {
    $env:SS_PTOOL_DOTNET
} else {
    (Get-Command dotnet -ErrorAction SilentlyContinue).Source
}
$ExistingBridge = Join-Path $BridgeOutput "CameraBridge.exe"
if ($DotnetExe -and (& $DotnetExe --list-sdks)) {
    & $DotnetExe build $BridgeProject -c Release
    if ($LASTEXITCODE -ne 0) {
        throw "CameraBridge 빌드에 실패했습니다. Exit code: $LASTEXITCODE"
    }
} elseif (Test-Path -LiteralPath $ExistingBridge) {
    Write-Warning ".NET 8 SDK가 없어 기존 CameraBridge Release 산출물을 사용합니다: $ExistingBridge"
} else {
    throw ".NET 8 SDK와 기존 CameraBridge Release 실행파일을 모두 찾을 수 없습니다."
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "SS_PTool_V0.1.5" `
    --paths (Join-Path $ProjectRoot "Source") `
    --collect-all matplotlib `
    --collect-all pyqtgraph `
    --hidden-import "scipy._external.array_api_compat.numpy.fft" `
    --hidden-import "scipy._external.array_api_compat.numpy.linalg" `
    --add-data "$BridgeOutput;CameraBridge" `
    --add-data "$CamDirectory;cam File" `
    (Join-Path $ProjectRoot "Source\main.py")

if ($LASTEXITCODE -ne 0) {
    throw "실행파일 빌드에 실패했습니다. Exit code: $LASTEXITCODE"
}

Write-Host "빌드 완료: $ProjectRoot\dist\SS_PTool_V0.1.5\SS_PTool_V0.1.5.exe"
