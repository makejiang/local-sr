# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

param(
    [Parameter(Position=0)][string]$InputFile,
    [Parameter(Position=1)][string]$OutputDir = '',
    [int]$Scale = 4
)

$ErrorActionPreference = 'Stop'

$LogDir = Join-Path $env:USERPROFILE '.openvino\log'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$LogFile = Join-Path $LogDir ("sr-client-$(Get-Date -Format 'yyyyMMdd-HHmmss').log")

if (-not $InputFile) { Write-Host 'Usage: scripts\run.ps1 "<input>" [output_dir] [-Scale 1|2|3|4]'; exit 1 }
if ($Scale -notin 1,2,3,4) { Write-Host 'Scale must be 1, 2, 3, or 4.'; exit 1 }

$PlatformScript = Join-Path $PSScriptRoot '..\bin\platform.ps1'
if (Test-Path $PlatformScript) {
    $IsAipc = & $PlatformScript --is-aipc
    if ($IsAipc -ne '1') { Write-Host 'This skill requires an Intel AIPC platform.'; exit 1 }
} else {
    Write-Host 'WARN: bin\platform.ps1 missing; skipping AIPC check.'
}

$SkillRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Push-Location $SkillRoot
try {
    $Config = Get-Content (Join-Path $SkillRoot 'info.json') -Raw | ConvertFrom-Json
    if (-not $Config.venv_name) { throw 'venv_name not found in info.json' }
    & (Join-Path $SkillRoot 'scripts\install-env.ps1') -SkillRoot $SkillRoot
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $VenvPy = Join-Path $env:USERPROFILE ".openvino\venv\$($Config.venv_name)\Scripts\python.exe"
    $ClientArgs = @('scripts\client.py', '-i', $InputFile, '--scale', [string]$Scale, '--log', $LogFile)
if ($OutputDir) { $ClientArgs += @('-o', $OutputDir) }
    & $VenvPy @ClientArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
