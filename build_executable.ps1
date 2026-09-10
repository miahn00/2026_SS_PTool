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
if (-not $DotnetExe -or -not (& $DotnetExe --list-sdks)) {
    throw ".NET 8 SDK를 찾을 수 없습니다. SDK 설치 후 SS_PTOOL_DOTNET에 dotnet.exe 경로를 지정할 수 있습니다."
}

& $DotnetExe build $BridgeProject -c Release
if ($LASTEXITCODE -ne 0) {
    throw "CameraBridge 빌드에 실패했습니다. Exit code: $LASTEXITCODE"
}

& $PythonExe -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name "SS_PTool_V0.1.4" `
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

Write-Host "빌드 완료: $ProjectRoot\dist\SS_PTool_V0.1.4\SS_PTool_V0.1.4.exe"
