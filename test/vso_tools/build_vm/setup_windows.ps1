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
$NugetDir = "C:\nuget"
New-Item "$NugetDir" -ItemType Directory -Force | Out-Null
Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile "${NugetDir}\nuget.exe"
$env:path = "$env:path;$NugetDir"

Write-Host "Installing utilities..."

# Install azcopy for cache download.
# Something wrong with the latest (10.15.0) checksum.
choco install -y --force azcopy10 --version=10.14.1
azcopy --version

# Install swig.
# Note that swig 4.0 is not compatible with ConfigSpace.
choco install -y --force swig --version=3.0.12
swig -version

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

# Install python.
Write-Host "Installing Python..."
$PythonDir = "C:\Python"
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.9.12/python-3.9.12-amd64.exe" -OutFile "python-installer.exe"
Start-Process -FilePath .\python-installer.exe -NoNewWindow -Wait -ArgumentList "/quiet InstallAllUsers=1 TargetDir=C:\Python\ Include_launcher=0"
dir $PythonDir
$env:path = "$env:path;$PythonDir"
Remove-Item python-installer.exe
Write-Host "Verify Python installation..."
python --version

# Here are some other comments, which might be useful when installing python without elevation.
#
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
#
# $PythonDir = "C:\Python"
# nuget install python -Version 3.9.12 -OutputDirectory "$PythonDir"
# $env:path = "$env:path;$PythonDir\python.3.9.12\tools\"

# Permanently update the PATHs
# https://codingbee.net/powershell/powershell-make-a-permanent-change-to-the-path-environment-variable
Write-Host "Prepare PATHs..."
Write-Host $env:path
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH -Value $env:path
