# 构建向量索引（药品知识库 AI 助手）
# 用法：
#   pwsh ./scripts/build.ps1
#   pwsh ./scripts/build.ps1 -Data D:/other/all_medicine.json
[CmdletBinding()]
param(
    [string]$Data = ""
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$env:PYTHONUTF8 = "1"

$pythonExe = "D:/soft/Miniconda/envs/lang/python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Warning "未找到 $pythonExe，回退到 PATH 中的 python"
    $pythonExe = "python"
}

$args = @()
if ($Data) { $args += @("--data", $Data) }

& $pythonExe build_index.py @args
