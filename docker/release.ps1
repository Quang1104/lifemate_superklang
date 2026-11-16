#Requires -Version 5.1
<#
.SYNOPSIS
Build and export the LifeMate image. Base planner and ASR weights are downloaded inside the build
(pinned revisions in configs/colab_models.json) and the QLoRA adapter is merged there, so the only
local input is the adapter folder (~130 MB).
A fine-tuned ASR (training_code/train_asr.py output) placed in <AdapterDir>\asr replaces
the pinned PhoWhisper download unchanged (medium or large; docker/fetch_models.py).
Large fp16 weights are approximately 3.1 GB (estimate; measure the actual checkpoint).
.EXAMPLE
powershell.exe -NoProfile -File .\docker\release.ps1 -CheckOnly
powershell.exe -NoProfile -File .\docker\release.ps1 -AdapterDir C:\lifemate_release\qlora_native
powershell.exe -NoProfile -File .\docker\release.ps1 -AdapterDir C:\lifemate_release\qlora_v4 -Flags "LIFEMATE_CONTEXT_V4=1,LIFEMATE_SMALL_FIXES=1"
-Flags names the feature flags the chosen planner was trained with (allowlist: docker\release_flags.py). Each becomes
an ENV line of the image and is recorded in <archive>.manifest.json with the adapter and ASR SHA256.
Undo everything this script created with docker\cleanup.ps1 (see docker\LOCAL_DISK_NOTES.md).
#>
[CmdletBinding()]
param(
    [string]$AdapterDir,
    [ValidateSet('asr', 'asr_large')]
    [string]$AsrRole = 'asr',  # fallback download only; <AdapterDir>\asr takes precedence
    [string]$ImageTag = 'lifemate:2026-11-15',
    # Outside the repo: the repo lives in OneDrive and a 15+ GB archive must not be synced.
    [string]$OutputArchive = (Join-Path $env:USERPROFILE 'lifemate_release\lifemate-2026-11-15.tar.gz'),
    # "NAME=1,NAME=1": feature flags baked into the image; empty keeps every flag off.
    [string]$Flags = '',
    [switch]$CheckOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
# Validate the flags before anything else: a planner run without its training-time flags scores far lower.
$flagLines = @(& python -B (Join-Path $PSScriptRoot 'release_flags.py') "flags:$Flags" | Where-Object { $_ })
if ($LASTEXITCODE -ne 0) { throw "Invalid -Flags '$Flags' (see docker\release_flags.py)." }
Write-Output "Release flags: $(if ($flagLines) { $flagLines -join ' ' } else { '(none)' })"
$stateFile = Join-Path $env:USERPROFILE 'lifemate_release\release-state.json'
# Estimate, not measured: image ~21 GB plus a compressed archive and transient download space.
$neededGB = 40
$dockerDrive = (Get-Item -LiteralPath $env:LOCALAPPDATA).PSDrive.Name
$freeGB = [math]::Round((Get-PSDrive -Name $dockerDrive).Free / 1GB, 1)
Write-Output "Free space on ${dockerDrive}: $freeGB GB; estimated peak need: $neededGB GB."
if ($CheckOnly) { exit 0 }
if ($freeGB -lt $neededGB) {
    throw "Need about $neededGB GB free on ${dockerDrive}: before building; found $freeGB GB."
}
if (-not $AdapterDir) { throw 'Pass -AdapterDir <folder with adapter_config.json and adapter_model.safetensors>.' }
$adapter = (Resolve-Path -LiteralPath $AdapterDir).Path
foreach ($name in @('adapter_config.json', 'adapter_model.safetensors')) {
    if (-not (Test-Path -LiteralPath (Join-Path $adapter $name) -PathType Leaf)) {
        throw "Adapter folder is missing $name"
    }
}
if (Test-Path -LiteralPath $OutputArchive) { throw "Refusing to overwrite $OutputArchive" }
& docker info --format '{{.ServerVersion}}' | Out-Null
if ($LASTEXITCODE -ne 0) { throw 'Docker is not running: start Docker Desktop first.' }

# Remember the images that already existed so cleanup.ps1 only removes what this release added.
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $OutputArchive) | Out-Null
$imagesBefore = @(& docker image ls --quiet --no-trunc)
@{ image = $ImageTag; archive = $OutputArchive; images_before = $imagesBefore
   started = (Get-Date).ToString('s') } | ConvertTo-Json | Set-Content -LiteralPath $stateFile -Encoding UTF8

$buildArguments = @('build', '--file', (Join-Path $PSScriptRoot 'Dockerfile'),
    '--build-context', "adapter=$adapter", '--tag', $ImageTag, '--build-arg', "ASR_ROLE=$AsrRole")
foreach ($flag in $flagLines) { $buildArguments += @('--build-arg', $flag) }
$buildArguments += $repo
& docker @buildArguments
if ($LASTEXITCODE -ne 0) { throw "docker build failed with exit code $LASTEXITCODE" }

# Stream `docker save` through .NET gzip: PowerShell 5.1 pipes corrupt binary data, and an
# intermediate .tar would double the disk needed.
$startInfo = New-Object System.Diagnostics.ProcessStartInfo 'docker', "save $ImageTag"
$startInfo.RedirectStandardOutput = $true
$startInfo.UseShellExecute = $false
$process = [System.Diagnostics.Process]::Start($startInfo)
$file = [System.IO.File]::Create($OutputArchive)
$gzip = New-Object System.IO.Compression.GZipStream($file, [System.IO.Compression.CompressionLevel]::Fastest)
try { $process.StandardOutput.BaseStream.CopyTo($gzip) } finally { $gzip.Dispose(); $file.Dispose() }
$process.WaitForExit()
if ($process.ExitCode -ne 0) {
    Remove-Item -LiteralPath $OutputArchive -Force
    throw "docker save failed with exit code $($process.ExitCode); no archive kept."
}
$md5 = (Get-FileHash -LiteralPath $OutputArchive -Algorithm MD5).Hash.ToLowerInvariant()
"$md5  $(Split-Path -Leaf $OutputArchive)" | Set-Content -LiteralPath "$OutputArchive.md5" -Encoding Ascii
# What this archive is: flags, adapter and (when supplied) fine-tuned ASR, so the image can be audited later.
$asrWeights = Join-Path $adapter 'asr\model.safetensors'
[ordered]@{
    image = $ImageTag; flags = $flagLines; archive_md5 = $md5
    adapter_sha256 = (Get-FileHash -LiteralPath (Join-Path $adapter 'adapter_model.safetensors') -Algorithm SHA256).Hash.ToLowerInvariant()
    asr_sha256 = $(if (Test-Path -LiteralPath $asrWeights -PathType Leaf) {
        (Get-FileHash -LiteralPath $asrWeights -Algorithm SHA256).Hash.ToLowerInvariant()
    } elseif (Test-Path -LiteralPath (Join-Path $adapter 'asr')) { 'asr folder without model.safetensors: not hashed'
    } else { 'pinned PhoWhisper download' })
    built = (Get-Date).ToString('s')
} | ConvertTo-Json | Set-Content -LiteralPath "$OutputArchive.manifest.json" -Encoding UTF8
$imageGB = [math]::Round([double](& docker image inspect $ImageTag --format '{{.Size}}') / 1GB, 2)
$archiveGB = [math]::Round((Get-Item -LiteralPath $OutputArchive).Length / 1GB, 2)
Write-Output "Image $($ImageTag): $imageGB GB. Archive: $OutputArchive ($archiveGB GB), MD5 $md5"
Write-Output 'Load it elsewhere with: docker load -i <archive>   (gzip archives load directly)'
