# Choco.
# https://docs.chocolatey.org/en-us/choco/setup
# Community version can't customize output directory.
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Nuget.
# Doesn't have azcopy.
$NugetDir = "C:\nuget"
New-Item "$NugetDir" -ItemType Directory -Force
Invoke-WebRequest -Uri "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe" -OutFile "${NugetDir}\nuget.exe"
$env:path = "$env:path;$NugetDir"

# Install azcopy for cache download.
# Something wrong with the latest (10.15.0) checksum.
choco install -y --force azcopy10 --version=10.14.1
azcopy --version

# Install swig.
# Note that swig 4.0 is not compatible with ConfigSpace.
choco install -y --force swig --version=3.0.12
swig -version

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
$PythonDir = "C:\Python"
nuget install python -Version 3.9.12 -OutputDirectory "$PythonDir"
$env:path = "$env:path;$PythonDir\python.3.9.12\tools\"

# Permanently update the PATHs
# https://codingbee.net/powershell/powershell-make-a-permanent-change-to-the-path-environment-variable
Write-Host $env:path
Set-ItemProperty -Path 'Registry::HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment' -Name PATH -Value $env:path
