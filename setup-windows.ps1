$ErrorActionPreference = 'Stop'

$python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    try {
        & py -3.12 -c 'import sys; assert sys.version_info[:2] == (3, 12)'
        if ($LASTEXITCODE -eq 0) { $python = @('py', '-3.12') }
    } catch {}
}
if ($null -eq $python -and (Get-Command python -ErrorAction SilentlyContinue)) {
    try {
        & python -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12)'
        if ($LASTEXITCODE -eq 0) { $python = @('python') }
    } catch {}
}
if ($null -eq $python) {
    throw 'CartSense requires Python 3.10–3.12 (64-bit). Install Python 3.12 and rerun this script.'
}

if ($python.Length -eq 1) {
    & $python[0] -m venv .venv
} else {
    & $python[0] $python[1] -m venv .venv
}
if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv.' }
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'Could not upgrade pip.' }
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Setup complete. Activate with: .\.venv\Scripts\Activate.ps1'
