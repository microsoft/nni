param ([Switch] $Development)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12


$install_node = $true
$install_yarn = $true

if ([Environment]::Is64BitOperatingSystem) {
    $OS_VERSION = 'win64'
    $nodeUrl = "https://nodejs.org/download/release/v10.22.1/node-v10.22.1-win-x64.zip"
}
else {
    $OS_VERSION = 'win32'
    $nodeUrl = "https://nodejs.org/download/release/v10.22.1/node-v10.22.1-win-x86.zip"
}
# nodejs
$yarnUrl = "https://github.com/yarnpkg/yarn/releases/download/v1.22.5/yarn-v1.22.5.tar.gz"
$unzipNodeDir = "node-v*"
$unzipYarnDir = "yarn-v*"

$NNI_DEPENDENCY_FOLDER = [System.IO.Path]::GetTempPath() + $env:USERNAME

$WHICH_PYTHON = where.exe python
if ($WHICH_PYTHON -eq $null) {
    throw "Can not find python"
}
else {
    $pyVersion = & python -V 2>&1
    $pyVersion = ([string]$pyVersion).substring(7, 3)
    if ([double]$pyVersion -lt 3.5) {
        throw "python version should >= 3.5"
    }
}

$WHICH_PIP = where.exe pip
if ($WHICH_PIP -eq $null) {
    throw "Can not find pip"
}

$env:PYTHONIOENCODING = "UTF-8"
if ($env:VIRTUAL_ENV) {
    $NNI_PYTHON3 = $env:VIRTUAL_ENV + "\Scripts"
    $NNI_PKG_FOLDER = $env:VIRTUAL_ENV + "\nni"
    $NNI_PYTHON_SCRIPTS = $NNI_PYTHON3
}
else {
    $NNI_PYTHON3 = $(python -c 'import site; from pathlib import Path; print(Path(site.getsitepackages()[0]))')
    $NNI_PKG_FOLDER = $NNI_PYTHON3 + "\nni"
    $NNI_PYTHON_SCRIPTS = $NNI_PYTHON3 + "\Scripts"
}

$PIP_INSTALL = """$NNI_PYTHON3\python"" -m pip install "

if (!(Test-Path $NNI_DEPENDENCY_FOLDER)) {
    $null = New-Item $NNI_DEPENDENCY_FOLDER -ItemType Directory
}
$NNI_NODE_ZIP = $NNI_DEPENDENCY_FOLDER + "\nni-node.zip"
$NNI_NODE_FOLDER = $NNI_DEPENDENCY_FOLDER + "\nni-node"
$NNI_YARN_TARBALL = $NNI_DEPENDENCY_FOLDER + "\nni-yarn.tar.gz"
$NNI_YARN_FOLDER = $NNI_DEPENDENCY_FOLDER + "\nni-yarn"
$NNI_YARN = $NNI_YARN_FOLDER + "\bin\yarn"

## Version number
$NNI_VERSION_VALUE = $(git describe --tags)
$NNI_VERSION_TEMPLATE = "999.0.0-developing"

if (!(Test-Path $NNI_NODE_ZIP)) {
    Write-Host "Downloading Node..."
    (New-Object Net.WebClient).DownloadFile($nodeUrl, $NNI_NODE_ZIP)
}

if (!(Test-Path $NNI_YARN_TARBALL)) {
    Write-Host "Downloading Yarn..."
    (New-Object Net.WebClient).DownloadFile($yarnUrl, $NNI_YARN_TARBALL)
}

$NNI_YARN_TARBALL = $NNI_YARN_TARBALL -split '\\' -join '\\'
$NNI_DEPENDENCY_FOLDER = $NNI_DEPENDENCY_FOLDER -split '\\' -join '\\'
$SCRIPT_PATH = $NNI_DEPENDENCY_FOLDER + '\extract.py'
$SCRIPT = "import tarfile",
        ("tar = tarfile.open(""{0}"")" -f $NNI_YARN_TARBALL),
        ("tar.extractall(""{0}"")" -f $NNI_DEPENDENCY_FOLDER),
        "tar.close()"
[System.IO.File]::WriteAllLines($SCRIPT_PATH, $SCRIPT)

Add-Type -AssemblyName System.IO.Compression.FileSystem
function Unzip {
    param([string]$zipfile, [string]$outpath)
    [System.IO.Compression.ZipFile]::ExtractToDirectory($zipfile, $outpath)
}
if ($install_node) {
    ### nodejs install
    if (!(Test-Path $NNI_NODE_FOLDER)) {
        Unzip $NNI_NODE_ZIP $NNI_DEPENDENCY_FOLDER
        $unzipNodeDir = Get-ChildItem "$NNI_DEPENDENCY_FOLDER\$unzipNodeDir"
        Rename-Item $unzipNodeDir "nni-node"
    }
    Copy-Item "$NNI_NODE_FOLDER\node.exe" $NNI_PYTHON_SCRIPTS -Recurse -Force
}

if ($install_yarn) {
    ### yarn install
    if (!(Test-Path $NNI_YARN_FOLDER)) {
        cmd /C """$NNI_PYTHON3\python""" $SCRIPT_PATH
        $unzipYarnDir = Get-ChildItem "$NNI_DEPENDENCY_FOLDER\$unzipYarnDir"
        Rename-Item $unzipYarnDir "nni-yarn"
    }
}

## install-python-modules:
### Installing Python SDK
(Get-Content setup.py).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content setup.py

if ($Development) {
    $PYTHON_BUILD = "build"
    # To compat with file and links.
    cmd /c if exist "$PYTHON_BUILD" rmdir /s /q $PYTHON_BUILD

    $null = New-Item $PYTHON_BUILD -ItemType Directory
    $null = New-Item -ItemType Junction -Path "$($PYTHON_BUILD)\nni" -Target "src\sdk\pynni\nni"
    $null = New-Item -ItemType Junction -Path "$($PYTHON_BUILD)\nnicli" -Target "src\sdk\pycli\nnicli"
    $null = New-Item -ItemType Junction -Path "$($PYTHON_BUILD)\nni_annotation" -Target "tools\nni_annotation"
    $null = New-Item -ItemType Junction -Path "$($PYTHON_BUILD)\nni_cmd" -Target "tools\nni_cmd"
    $null = New-Item -ItemType Junction -Path "$($PYTHON_BUILD)\nni_trial_tool" -Target "tools\nni_trial_tool"
    $null = New-Item -ItemType Junction -Path "$($PYTHON_BUILD)\nni_gpu_tool" -Target "tools\nni_gpu_tool"

    Copy-Item setup.py $PYTHON_BUILD
    Copy-Item README.md $PYTHON_BUILD

    Push-Location build
    #update folders in setup file
    (Get-Content setup.py).replace("src/sdk/pynni/", "") | Set-Content setup.py
    (Get-Content setup.py).replace("src/sdk/pycli/", "") | Set-Content setup.py
    (Get-Content setup.py).replace("src/sdk/pynni", ".") | Set-Content setup.py
    (Get-Content setup.py).replace("tools/", "") | Set-Content setup.py
    # install current folder.
    cmd /c $PIP_INSTALL -e .
    Pop-Location
}
else {
    cmd /c $PIP_INSTALL .
}

# Building NNI Manager
$env:PATH = $NNI_PYTHON_SCRIPTS + ';' + $env:PATH
cd src\nni_manager
cmd /c $NNI_YARN
cmd /c $NNI_YARN build
Copy-Item config -Destination .\dist\ -Recurse -Force
# Building WebUI
# office-ui-fabric-react need longer time. the 180000 is in ms, mean 180 seconds, longer than default 30 seconds.
cd ..\webui
cmd /c $NNI_YARN --network-timeout 180000
cmd /c $NNI_YARN build
# Building NasUI
cd ..\nasui
cmd /c $NNI_YARN --network-timeout 180000
cmd /c $NNI_YARN build

cd ..\..

## install-node-modules

# it needs to remove the whole folder for following copy.
cmd /c if exist "$NNI_PKG_FOLDER" rmdir /s /q $NNI_PKG_FOLDER

$NNI_PKG_FOLDER_STATIC = $NNI_PKG_FOLDER + "\static"
$NASUI_PKG_FOLDER = $NNI_PKG_FOLDER + "\nasui"

cmd /c if exist "src\nni_manager\dist\node_modules" rmdir /s /q src\nni_manager\dist\node_modules
cmd /c if exist "src\nni_manager\dist\static" rmdir /s /q src\nni_manager\dist\static
cmd /c if exist "src\nni_manager\dist\nasui" rmdir /s /q src\nni_manager\dist\nasui

if ($Development) {
    $null = New-Item -ItemType Junction -Path $NNI_PKG_FOLDER -Target "src\nni_manager\dist"

    $null = New-Item -ItemType Junction -Path "$($NNI_PKG_FOLDER)\node_modules" -Target "src\nni_manager\node_modules"
    $null = New-Item -ItemType Junction -Path $NNI_PKG_FOLDER_STATIC -Target "src\webui\build"
    $null = New-Item -ItemType Junction -Path $NASUI_PKG_FOLDER -Target "src\nasui\build"
}
else {
    Copy-Item "src\nni_manager\dist" $NNI_PKG_FOLDER -Recurse
    Copy-Item "src\webui\build" $NNI_PKG_FOLDER_STATIC -Recurse
    Copy-Item "src\nasui\build" $NASUI_PKG_FOLDER -Recurse

    Copy-Item "src\nni_manager\package.json" $NNI_PKG_FOLDER
    $PKG_JSON = $NNI_PKG_FOLDER + "\package.json"
    (Get-Content $PKG_JSON).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content $PKG_JSON

    cmd /c $NNI_YARN --prod --cwd $NNI_PKG_FOLDER
}

Copy-Item "src\nasui\server.js" $NASUI_PKG_FOLDER
