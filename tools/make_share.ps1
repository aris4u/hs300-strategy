# Build a zip others can unzip and double-click.
# Output: <project>\发给别人\HS300选股.zip
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
if (Test-Path "C:\Users\$env:USERNAME\HS300\HS300.cmd") {
    $Root = "C:\Users\$env:USERNAME\HS300"
}
Set-Location $Root
Write-Host "pack from $Root"

$sysPy = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
if (-not (Test-Path $sysPy)) { $sysPy = "python" }
& $sysPy (Join-Path $Root "tools\ensure_runtime.py") --runtime
if ($LASTEXITCODE -ne 0) { throw "ensure_runtime failed" }
if (-not (Test-Path (Join-Path $Root "runtime\python.exe"))) {
    throw "runtime\python.exe missing; share zip would not run on other PCs"
}

$stage = Join-Path $env:TEMP "HS300-share"
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
New-Item -ItemType Directory -Path $stage | Out-Null

$excludeDir = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
foreach ($n in @(".venv", "__pycache__", "发给别人", "tools\_cache", ".git")) { [void]$excludeDir.Add($n) }

function Copy-Tree($src, $dst) {
    New-Item -ItemType Directory -Force -Path $dst | Out-Null
    Get-ChildItem -LiteralPath $src -Force | ForEach-Object {
        if ($excludeDir.Contains($_.Name)) { return }
        if ($_.Name -eq ".env") { return }
        if ($_.Name -eq "universe_v1.pkl") { return }
        $target = Join-Path $dst $_.Name
        if ($_.PSIsContainer) {
            Copy-Tree $_.FullName $target
        } else {
            Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        }
    }
}

Copy-Tree $Root $stage
$outDir = Join-Path $Root "发给别人"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$zip = Join-Path $outDir "HS300选股.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Push-Location $stage
tar -a -c -f $zip *
Pop-Location
Remove-Item $stage -Recurse -Force
Write-Host "OK $zip"
Write-Host ("size MB " + [math]::Round((Get-Item $zip).Length / 1MB, 1))
