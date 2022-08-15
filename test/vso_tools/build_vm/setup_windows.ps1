#Requires -RunAsAdministrator
$ErrorActionPreference = "Stop"

# Choco.
# https://docs.chocolatey.org/en-us/choco/setup
# Community version can't customize output directory.
Write-Host "Installing Choco..."
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

Set-PSDebug -Trace 1

# Nuget.
# Doesn't have azcopy.
Write-Host "Installing Nuget..."
$NugetDir = "$env:ProgramData\nuget"
New-Item "$NugetDir" -ItemType Directory -Force | Out-Null
Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile "${NugetDir}\nuget.exe"
$env:path = "$env:path;$NugetDir"

# Install SSH.
Write-Host "Installing SSH..."
# https://docs.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH*'
# Install the OpenSSH Client
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
# Install the OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
# Start the sshd service
Set-PSDebug -Trace 0
Write-Host "Starting SSH service..."
Start-Service sshd
Set-Service -Name sshd -StartupType 'Automatic'
Write-Host "Configure firewall for SSH..."
# Confirm the Firewall rule is configured. It should be created automatically by setup. Run the following to verify
if (!(Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue | Select-Object Name, Enabled)) {
    Write-Output "Firewall Rule 'OpenSSH-Server-In-TCP' does not exist, creating it..."
    New-NetFirewallRule -Name 'OpenSSH-Server-In-TCP' -DisplayName 'OpenSSH Server (sshd)' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
} else {
    Write-Output "Firewall rule 'OpenSSH-Server-In-TCP' has been created and exists."
}

Set-PSDebug -Trace 1

# Create a new user (for SSH login).
$Password = ConvertTo-SecureString "P@ssW0rD!" -AsPlainText -Force
New-LocalUser "NNIUser" -Password $Password -PasswordNeverExpires

# These installation seems not working.

# Visual Studio C++ Build tools (for Cython)
# Invoke-WebRequest "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile "vs_BuildTools.exe"
# Start-Process -FilePath "vs_BuildTools.exe" -ArgumentList "--quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" -Wait
# Remove-Item "vs_BuildTools.exe"

# Microsoft Visual C++ Redistributable (for PyTorch)
# Invoke-WebRequest "https://aka.ms/vs/16/release/vc_redist.x64.exe" -OutFile "vc_redist.x64.exe"
# Start-Process -FilePath ".\vc_redist.x64.exe" -ArgumentList "/q /norestart" -Wait
# Remove-Item "vc_redist.x64.exe"

# Use choco instead.
choco install -y --no-progress visualstudio2019buildtools
choco install -y --no-progress visualstudio2019-workload-vctools
choco install -y --no-progress vcredist2012 vcredist2013 vcredist2015 vcredist2017

# Install CUDA.
Write-Host "Installing CUDA..."
$CudaUrl = "https://developer.download.nvidia.com/compute/cuda/11.7.0/network_installers/cuda_11.7.0_windows_network.exe"
Invoke-WebRequest $CudaUrl -OutFile "$env:ProgramData\cuda_installer.exe"
Start-Process -FilePath "$env:ProgramData\cuda_installer.exe" -ArgumentList "/s /n" -Wait
# Remove-Item "cuda_installer.exe"
# Verify CUDA.
Write-Host "Verify CUDA installation..."
$CudaDir = "$env:ProgramFiles\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin"
# GPU driver can't be installed without a hardware
# Get-Command nvidia-smi
Get-ChildItem $CudaDir
$env:path = "$env:path;$CudaDir"

# Download GPU driver.
Invoke-WebRequest "https://us.download.nvidia.com/tesla/516.94/516.94-data-center-tesla-desktop-winserver-2016-2019-2022-dch-international.exe" -OutFile "$env:ProgramData\driver_installer.exe"

Write-Host "Installing utilities..."

# Install azcopy for cache download.
# Something wrong with the latest (10.15.0) checksum.
choco install -y --force azcopy10 --version=10.14.1 --no-progress
azcopy --version

# Install swig.
# Note that swig 4.0 is not compatible with ConfigSpace.
choco install -y --force swig --version=3.0.12 --no-progress
swig -version

# Install cmake.
choco install -y --no-progress cmake
$env:path = "$env:path;$env:ProgramFiles\CMake\bin"
cmake --version

# Install python.
# Originally I tried to install the python by downloading from official, and run the installation.
#
#     Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.9.12/python-3.9.12-amd64.exe" -OutFile "python-installer.exe"
#     Start-Process -FilePath .\python-installer.exe -NoNewWindow -Wait \
#       -ArgumentList "/quiet InstallAllUsers=1 TargetDir=$(Agent.ToolsDirectory)\Python\3.9.12\x64 Include_launcher=0"
#     New-Item -Path $(Agent.ToolsDirectory)\Python\3.9.12\x64.complete -ItemType file -Force
#
# But ``Start-Process`` fails with mysterious reasons (exit code is not zero and no error message).
# I tried with -PassThru, -NoNewWindow, -Wait, /quiet, /passive, InstallAllUsers and some other flags, but none works.
# (InstallAllUsers is the key to make it work on my local, but not on pipeline).
# I guess it's related to lack of adminstrative privileges.
# I kept this attempt here in case any one can make it work.
#
# Other two workarounds.
# 1) choco install python. The community verison can't customize output directory,
#    and the output directory is only a guess (e.g., C:\Python310).
# 2) nuget install python. This seems working.
#
# Can't move to the installed python to $PythonDir\3.9.12\x64 because,
# 1. If we copy it, Python path will complain in the next few steps.
# 2. If we try to create a symlink, it will tell us that we don't have adminstrative rights.
#
# After all this struggle, the workaround here is simple:
# to install with nuget, then don't use `UsePythonVersion` in the next step.
# The workaround works because we actually never needs multiple python versions on windows.
Write-Host "Installing Python..."
$PythonDir = "$env:ProgramData\Python"
nuget install python -Version 3.9.12 -OutputDirectory "$PythonDir"
$env:path = "$env:path;$PythonDir\python.3.9.12\tools\;$PythonDir\python.3.9.12\tools\Scripts"
Write-Host "Verify Python installation..."
python --version

# Permanently update the PATHs
# https://codingbee.net/powershell/powershell-make-a-permanent-change-to-the-path-environment-variable
Write-Host "Prepare PATHs..."
Write-Host $env:path
Set-ItemProperty -Path "Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment" -Name PATH -Value $env:path
