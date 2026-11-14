#Requires -Version 5.1
<#
.SYNOPSIS
Build and export the LifeMate release for 2026-11-15.
.EXAMPLE
powershell.exe -NoProfile -File .\docker\release.ps1 -PlannerDir C:\models\planner-merged -AsrDir C:\models\phowhisper-medium
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$PlannerDir,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$AsrDir,

    [ValidateNotNullOrEmpty()]
    [string]$ImageTag = 'lifemate:2026-11-15',

    [ValidateNotNullOrEmpty()]
    [string]$OutputTar = (Join-Path $PSScriptRoot 'lifemate-2026-11-15.tar')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-ModelDirectory {
    param(
        [string]$Path,
        [string]$Label
    )

    # Never traverse model trees; the selected directory is the build context.
    if ($Path -split '[\\/]' | Where-Object { $_ -eq 'state.json' -or $_ -like '*_environment' }) {
        throw "$Label path contains a forbidden component."
    }
    $resolved = (Resolve-Path -LiteralPath $Path -ErrorAction Stop).ProviderPath
    if ($resolved -split '[\\/]' | Where-Object { $_ -eq 'state.json' -or $_ -like '*_environment' }) {
        throw "$Label path contains a forbidden component."
    }
    if (-not (Test-Path -LiteralPath $resolved -PathType Container)) {
        throw "$Label must be a directory: $resolved"
    }
    $config = Join-Path $resolved 'config.json'
    if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
        throw "$Label lacks config.json: $resolved"
    }
    $weights = @(Get-ChildItem -LiteralPath $resolved -File -Filter '*.safetensors')
    if ($weights.Count -eq 0) {
        throw "$Label lacks safetensors weights: $resolved"
    }
    return $resolved
}

$plannerPath = Resolve-ModelDirectory -Path $PlannerDir -Label 'Merged planner'
$asrPath = Resolve-ModelDirectory -Path $AsrDir -Label 'ASR'
$repoRoot = Split-Path -Parent $PSScriptRoot
$dockerfile = Join-Path $PSScriptRoot 'Dockerfile'
$tarPath = [System.IO.Path]::GetFullPath($OutputTar)
if ([System.IO.Path]::GetExtension($tarPath) -ine '.tar') {
    throw 'OutputTar must end in .tar.'
}
$md5Path = "$tarPath.md5"
if ((Test-Path -LiteralPath $tarPath) -or (Test-Path -LiteralPath $md5Path)) {
    throw "Refusing to overwrite an existing archive or checksum: $tarPath"
}
Get-Command docker -CommandType Application -ErrorAction Stop | Out-Null
$tarParent = Split-Path -Parent $tarPath
if (-not (Test-Path -LiteralPath $tarParent -PathType Container)) {
    New-Item -ItemType Directory -Path $tarParent -ErrorAction Stop | Out-Null
}

$buildArguments = @(
    'build',
    '--file', $dockerfile,
    '--build-context', "planner_weights=$plannerPath",
    '--build-context', "asr_weights=$asrPath",
    '--tag', $ImageTag,
    $repoRoot
)
& docker @buildArguments
if ($LASTEXITCODE -ne 0) {
    throw "docker build failed with exit code $LASTEXITCODE."
}

& docker save --output $tarPath $ImageTag
if ($LASTEXITCODE -ne 0) {
    throw "docker save failed with exit code $LASTEXITCODE; no checksum was written."
}
if (-not (Test-Path -LiteralPath $tarPath -PathType Leaf)) {
    throw 'docker save did not create the archive; no checksum was written.'
}

$checksum = Get-FileHash -LiteralPath $tarPath -Algorithm MD5
$checksumLine = '{0}  {1}' -f $checksum.Hash.ToLowerInvariant(), [System.IO.Path]::GetFileName($tarPath)
Set-Content -LiteralPath $md5Path -Value $checksumLine -Encoding Ascii

$imageSizeOutput = & docker image inspect --format '{{.Size}}' $ImageTag
if ($LASTEXITCODE -ne 0) {
    throw "docker image inspect failed with exit code $LASTEXITCODE."
}
[long]$imageBytes = 0
if (-not [long]::TryParse(($imageSizeOutput -join '').Trim(), [ref]$imageBytes)) {
    throw 'docker image inspect returned an invalid image size.'
}
Write-Output ("Image: {0}" -f $ImageTag)
Write-Output ("Image size: {0} bytes ({1:N2} GiB)" -f $imageBytes, ($imageBytes / 1GB))
Write-Output ("Archive: {0}" -f $tarPath)
Write-Output ("MD5: {0}" -f $checksum.Hash.ToLowerInvariant())
Write-Output ("Checksum file: {0}" -f $md5Path)
