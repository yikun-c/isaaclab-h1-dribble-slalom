[CmdletBinding()]
param(
    [ValidateRange(4096, 32768)]
    [int]$SizeMiB = 16384
)

$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw 'Run this script from an elevated Administrator PowerShell window. No registry change was made.'
}

$memoryKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management'
$computer = Get-CimInstance Win32_ComputerSystem
if ($computer.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $computer -Property @{AutomaticManagedPagefile = $false} | Out-Null
}

$currentEntries = @((Get-ItemProperty -LiteralPath $memoryKey -Name PagingFiles).PagingFiles)
$dEntry = "D:\pagefile.sys $SizeMiB $SizeMiB"
$updatedEntries = @($currentEntries | Where-Object { $_ -notmatch '^D:\\pagefile\.sys\s' }) + $dEntry
Set-ItemProperty -LiteralPath $memoryKey -Name PagingFiles -Type MultiString -Value $updatedEntries

Write-Output 'Configured the D: page-file entry. Restart Windows before retrying Qwen or Isaac Lab.'
Get-ItemProperty -LiteralPath $memoryKey -Name PagingFiles | Select-Object -ExpandProperty PagingFiles
