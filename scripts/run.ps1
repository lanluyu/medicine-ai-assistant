# 启动 FastAPI 开发服务（药品知识库 AI 助手）
# 用法：
#   pwsh ./scripts/run.ps1                  # 默认 reload 模式
#   pwsh ./scripts/run.ps1 -NoReload        # 关闭 reload
#   pwsh ./scripts/run.ps1 -Port 8030       # 覆盖端口
[CmdletBinding()]
param(
    [switch]$NoReload,
    [string]$BindHost = "",
    [int]$Port = 0
)

$ErrorActionPreference = "Stop"

# 切到项目根目录（脚本所在目录的上一级）
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# 强制 UTF-8（避免 Windows GBK 干扰中文日志）
$env:PYTHONUTF8 = "1"

# 若调用方未指定 host/port，则交由 config.py 从 .env 读取默认值
$envArgs = @()
if ($BindHost) { $envArgs += @("--host", $BindHost) }
if ($Port -gt 0) { $envArgs += @("--port", "$Port") }

$reloadArg = if ($NoReload) { @() } else { @("--reload") }

# 默认走 conda 环境 lang（与 CLAUDE.md 全局约定一致）
$pythonExe = "D:/soft/Miniconda/envs/lang/python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Warning "未找到 $pythonExe，回退到 PATH 中的 python"
    $pythonExe = "python"
}

& $pythonExe -m uvicorn app_fastapi:app @reloadArg @envArgs
