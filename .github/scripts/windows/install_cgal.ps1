$ErrorActionPreference = "Stop"

$RepoRoot = $env:GITHUB_WORKSPACE
if ([string]::IsNullOrEmpty($RepoRoot)) {
  $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
}
$DepsRoot = Join-Path $RepoRoot ".deps"
$VcpkgRoot = "C:\vcpkg"
$Triplet = "x64-windows"
$InstalledRoot = Join-Path $VcpkgRoot "installed\$Triplet"
$CGALDir = Join-Path $InstalledRoot "share\cgal"
$GmpLib = Join-Path $InstalledRoot "lib\gmp.lib"
$MpfrLib = Join-Path $InstalledRoot "lib\mpfr.lib"
$BoostIncludeDir = Join-Path $InstalledRoot "include"

New-Item -ItemType Directory -Force -Path $DepsRoot | Out-Null

if (-not (Test-Path $VcpkgRoot)) {
  throw "vcpkg not found at $VcpkgRoot"
}

Push-Location $VcpkgRoot
try {
  Write-Host "Updating vcpkg..."
  & git pull
  if ($LASTEXITCODE -ne 0) { throw "git pull failed" }

  Write-Host "Bootstrapping vcpkg..."
  & .\bootstrap-vcpkg.bat
  if ($LASTEXITCODE -ne 0) { throw "bootstrap-vcpkg failed" }

  # CGAL's vcpkg build expects yasm-tool to be present beforehand.
  Write-Host "Installing yasm-tool for vcpkg..."
  & .\vcpkg install yasm-tool:x86-windows
  if ($LASTEXITCODE -ne 0) { throw "vcpkg install yasm-tool failed" }

  Write-Host "Installing CGAL and its dependencies via vcpkg..."
  & .\vcpkg install cgal --triplet $Triplet
  if ($LASTEXITCODE -ne 0) { throw "vcpkg install cgal failed" }
}
finally {
  Pop-Location
}

if (-not (Test-Path $CGALDir)) { throw "CGAL_DIR not found: $CGALDir" }
if (-not (Test-Path $GmpLib)) { throw "GMP library not found: $GmpLib" }
if (-not (Test-Path $MpfrLib)) { throw "MPFR library not found: $MpfrLib" }
if (-not (Test-Path $BoostIncludeDir)) { throw "Boost include dir not found: $BoostIncludeDir" }

Set-Content -Path (Join-Path $DepsRoot "cgal_dir.txt") -Value $CGALDir -Encoding ASCII
Set-Content -Path (Join-Path $DepsRoot "gmp_lib.txt") -Value $GmpLib -Encoding ASCII
Set-Content -Path (Join-Path $DepsRoot "mpfr_lib.txt") -Value $MpfrLib -Encoding ASCII
Set-Content -Path (Join-Path $DepsRoot "boost_include_dir.txt") -Value $BoostIncludeDir -Encoding ASCII

Write-Host "CGAL_DIR = $CGALDir"
