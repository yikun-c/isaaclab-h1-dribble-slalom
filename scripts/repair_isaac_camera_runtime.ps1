<#!
.SYNOPSIS
Repairs only the Isaac Lab virtual environment dependencies needed by the
RTX camera recorder. It does not alter system drivers, global PATH, or Isaac.
#>

[CmdletBinding()]
param(
    [string]$IsaacVenv = "D:\IsaacLab\.venv"
)

$python = Join-Path $IsaacVenv "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Isaac Lab Python was not found: $python"
}

# Isaac's generic_model_output plugin ships HDF5 1.14.6. h5py 3.15.0 is built
# against the same ABI; h5py 3.16.0 was built against HDF5 2.0 and caused
# generic_mo_io.dll to fail at Kit startup. The legacy tbb DLL is likewise
# required only inside this venv by that native plugin.
& $python -m pip install --force-reinstall --no-deps "tbb==2020.3.254" "h5py==3.15.0" --index-url https://pypi.org/simple
if ($LASTEXITCODE -ne 0) { throw "pip repair failed with exit code $LASTEXITCODE" }

& $python -c "import h5py; assert h5py.version.hdf5_version == '1.14.6', h5py.version.hdf5_version; print('camera-runtime-ready h5py=' + h5py.__version__ + ' hdf5=' + h5py.version.hdf5_version)"
if ($LASTEXITCODE -ne 0) { throw "HDF5 ABI verification failed" }

$tbb = Join-Path $IsaacVenv "Library\bin\tbb.dll"
if (-not (Test-Path -LiteralPath $tbb)) { throw "Missing expected venv DLL: $tbb" }
Write-Host "Camera runtime dependency repair verified."
