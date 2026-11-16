#Requires -Version 5.1
<#
.SYNOPSIS
Give back the disk space used by docker\release.ps1.
.DESCRIPTION
Removes the LifeMate image, every image the build added (for example the pulled CUDA base) but not
images that existed before the build, and Docker's build cache. Optional switches:
  -DeleteArchive  also delete the exported .tar.gz/.md5 (only after it has been uploaded)
  -DeleteAdapter  also delete the adapter folder downloaded for the build
  -CompactDisk    shrink Docker's WSL virtual disk so Windows gets the space back (run as Administrator;
                  quits Docker Desktop and shuts WSL down)
.EXAMPLE
powershell.exe -NoProfile -File .\docker\cleanup.ps1 -DeleteArchive -DeleteAdapter -CompactDisk
#>
[CmdletBinding()]
param([switch]$DeleteArchive, [switch]$DeleteAdapter, [switch]$CompactDisk)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$releaseDir = Join-Path $env:USERPROFILE 'lifemate_release'
$stateFile = Join-Path $releaseDir 'release-state.json'
if (-not (Test-Path -LiteralPath $stateFile)) { throw "No ${stateFile}: nothing recorded by release.ps1." }
$state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
$before = @($state.images_before)

# PS 5.1 turns a native command's stderr into terminating errors under 'Stop' (e.g. removing an
# image that is already gone), so Docker calls run with 'Continue' and are judged by exit code.
$ErrorActionPreference = 'Continue'
& docker info --format '{{.ServerVersion}}' *> $null
if ($LASTEXITCODE -eq 0) {
    & docker image rm --force $state.image *> $null
    foreach ($pass in 1..2) {  # a base image (lifemate:deps) can only go once the image built on it is gone
        foreach ($id in @(& docker image ls --quiet --no-trunc)) {
            if ($before -notcontains $id) { & docker image rm --force $id *> $null }
        }
    }
    & docker builder prune --all --force *> $null
    Write-Output 'Removed the LifeMate image, images added by the build, and the build cache.'
} else {
    Write-Output 'Docker is not running: start Docker Desktop and rerun to remove images.'
}
$ErrorActionPreference = 'Stop'

if ($DeleteArchive) {
    foreach ($path in @($state.archive, "$($state.archive).md5")) {
        if (Test-Path -LiteralPath $path) { Remove-Item -LiteralPath $path -Force; Write-Output "Deleted $path" }
    }
}
if ($DeleteAdapter) {
    $adapter = Join-Path $releaseDir 'qlora_native'
    if (Test-Path -LiteralPath $adapter) { Remove-Item -LiteralPath $adapter -Recurse -Force; Write-Output "Deleted $adapter" }
}
if ($CompactDisk) {
    $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $admin) { throw '-CompactDisk needs an Administrator PowerShell.' }
    $vhdx = Join-Path $env:LOCALAPPDATA 'Docker\wsl\disk\docker_data.vhdx'
    $beforeGB = [math]::Round((Get-Item -LiteralPath $vhdx).Length / 1GB, 1)
    $ErrorActionPreference = 'Continue'  # native wsl/diskpart; success is judged by the file size below
    # Without discarding freed blocks first, compact leaves the file at full size (measured 55.1 -> 55.1 GB).
    & wsl -d docker-desktop -- fstrim /mnt/docker-desktop-disk
    Get-Process -Name 'Docker Desktop' -ErrorAction SilentlyContinue | Stop-Process -Force
    & wsl --shutdown
    $script = Join-Path $env:TEMP 'lifemate-compact.txt'
    "select vdisk file=`"$vhdx`"`r`nattach vdisk readonly`r`ncompact vdisk`r`ndetach vdisk" |
        Set-Content -LiteralPath $script -Encoding Ascii
    & diskpart /s $script | Out-Null
    Remove-Item -LiteralPath $script -Force
    $ErrorActionPreference = 'Stop'
    $afterGB = [math]::Round((Get-Item -LiteralPath $vhdx).Length / 1GB, 1)
    Write-Output "Docker disk image: $beforeGB GB -> $afterGB GB"
}
if ($DeleteArchive -and $DeleteAdapter) { Remove-Item -LiteralPath $stateFile -Force }
