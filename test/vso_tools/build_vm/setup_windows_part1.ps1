#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

Set-PSDebug -Trace 1

# Visual Studio C++ Build tools (for Cython)
Invoke-WebRequest "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile "vs_BuildTools.exe"
Start-Process -FilePath "vs_BuildTools.exe" -ArgumentList "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" -Wait
Remove-Item "vs_BuildTools.exe"

# Microsoft Visual C++ Redistributable (for PyTorch)
Invoke-WebRequest "https://aka.ms/vs/16/release/vc_redist.x64.exe" -OutFile "vc_redist.x64.exe"
Start-Process -FilePath ".\vc_redist.x64.exe" -ArgumentList "/q /norestart" -Wait
Remove-Item "vc_redist.x64.exe"
