param([string]$profile = "release")

$ErrorActionPreference = "Stop"

$workspaceRoot = $PSScriptRoot
$crateName = "vk-debug-overlay"
$dllName = "VkLayer_planitia_debug_overlay.dll"
$manifestName = "VkLayer_planitia_debug_overlay.json"

# Map the cargo profile name to its output directory under `target/`.
# The built-in "dev" profile builds into "target/debug".
$targetDir = if ($profile -eq "dev") { "debug" } else { $profile }

Write-Host "Building $crateName (profile: $profile)..."
cargo build -p $crateName --profile $profile
if (-not $?) {
    throw "cargo build failed"
}

$builtDll = Join-Path $workspaceRoot "target\$targetDir\$dllName"
if (-not (Test-Path $builtDll)) {
    throw "Built layer DLL not found at $builtDll"
}

$binDir = Join-Path $workspaceRoot "bin"
New-Item -ItemType Directory -Path $binDir -Force | Out-Null

Write-Host "Copying $dllName to $binDir..."
Copy-Item -Path $builtDll -Destination $binDir -Force

$manifestSource = Join-Path $workspaceRoot "crates\vk-debug-overlay\$manifestName"
$manifestDest = Join-Path $binDir $manifestName
$dllDest = Join-Path $binDir $dllName

Write-Host "Writing $manifestName to $binDir..."
$manifestJson = Get-Content -Path $manifestSource -Raw | ConvertFrom-Json
$manifestJson.layer.library_path = $dllDest
$manifestJson | ConvertTo-Json -Depth 10 | Set-Content -Path $manifestDest -Encoding utf8

# Register the layer manifest with the Vulkan loader for the current user
# (no admin rights required), so `vkEnumerateInstanceLayerProperties` picks it up.
$registryKey = "HKCU:\Software\Khronos\Vulkan\ExplicitLayers"
New-Item -Path $registryKey -Force | Out-Null
New-ItemProperty -Path $registryKey -Name $manifestDest -Value 0 -PropertyType DWord -Force | Out-Null

Write-Host "Registered layer manifest: $manifestDest"
Write-Host "Done."