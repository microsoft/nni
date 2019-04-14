$install_node = $true
$install_yarn = $true

# nodejs
$nodeUrl = "https://aka.ms/nni/nodejs-download/win64"
$yarnUrl = "https://yarnpkg.com/latest.tar.gz"
$unzipNodeDir = "node-v*"
$unzipYarnDir = "yarn-v*"

$NNI_DEPENDENCY_FOLDER = "C:\tmp\$env:USERNAME"

$WHICH_PYTHON = where.exe python
if($WHICH_PYTHON -eq $null){
    throw "Can not find python"
}
else{
    $pyVersion = & python -V 2>&1
    $pyVersion =  ([string]$pyVersion).substring(7,3)
    if([double]$pyVersion -lt 3.5){
        throw "python version should >= 3.5"
    }
}

$WHICH_PIP = where.exe pip
if($WHICH_PIP -eq $null){
    throw "Can not find pip"
}
if($WHICH_PYTHON[0].Length -eq 1){
    $NNI_PYTHON3 = $WHICH_PYTHON.SubString(0,$WHICH_PYTHON.Length-11)
}
else{
    $NNI_PYTHON3 = $WHICH_PYTHON[0].SubString(0,$WHICH_PYTHON[0].Length-11)
}
$NNI_PKG_FOLDER = $NNI_PYTHON3 + "\nni"
$PIP_INSTALL = """$NNI_PYTHON3\python"" -m pip install ."

if(!(Test-Path $NNI_DEPENDENCY_FOLDER)){
    New-Item $NNI_DEPENDENCY_FOLDER -ItemType Directory
}
$NNI_NODE_ZIP = $NNI_DEPENDENCY_FOLDER+"\nni-node.zip"
$NNI_NODE_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-node"
$NNI_YARN_TARBALL = $NNI_DEPENDENCY_FOLDER+"\nni-yarn.tar.gz"
$NNI_YARN_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-yarn"
$NNI_YARN = $NNI_YARN_FOLDER +"\bin\yarn"

## Version number
$NNI_VERSION_VALUE = $(git describe --tags)
$NNI_VERSION_TEMPLATE = "999.0.0-developing"

if(!(Test-Path $NNI_NODE_ZIP)){
    Write-Host "Downloading Node..."
    (New-Object Net.WebClient).DownloadFile($nodeUrl, $NNI_NODE_ZIP)
}

if(!(Test-Path $NNI_YARN_TARBALL)){
    Write-Host "Downloading Yarn..."
    (New-Object Net.WebClient).DownloadFile($yarnUrl, $NNI_YARN_TARBALL)
}

$NNI_YARN_TARBALL = $NNI_YARN_TARBALL -split '\\' -join '\\'
$NNI_DEPENDENCY_FOLDER = $NNI_DEPENDENCY_FOLDER -split '\\' -join '\\'
$SCRIPT_PATH = $NNI_DEPENDENCY_FOLDER + '\extract.py'
$SCRIPT =  "import tarfile",
       ("tar = tarfile.open(""{0}"")" -f $NNI_YARN_TARBALL),
       ("tar.extractall(""{0}"")" -f $NNI_DEPENDENCY_FOLDER),
        "tar.close()"
[System.IO.File]::WriteAllLines($SCRIPT_PATH, $SCRIPT)

if ($install_node) {
    ### nodejs install
    if(!(Test-Path $NNI_NODE_FOLDER)){
        Expand-Archive $NNI_NODE_ZIP -DestinationPath $NNI_DEPENDENCY_FOLDER
        $unzipNodeDir = Get-ChildItem "$NNI_DEPENDENCY_FOLDER\$unzipNodeDir"
        Rename-Item $unzipNodeDir "nni-node"
    }

    ### yarn install
    if(!(Test-Path $NNI_YARN_FOLDER)){
        cmd /C "$NNI_PYTHON3\python" $SCRIPT_PATH
        $unzipYarnDir = Get-ChildItem "$NNI_DEPENDENCY_FOLDER\$unzipYarnDir"
        Rename-Item $unzipYarnDir "nni-yarn"
    }
}

### add to PATH
function Add2Path {
    param ($fileName)
    $PathVariable = [System.Environment]::GetEnvironmentVariable("Path","User")
    $PathFolders = $PathVariable.Split(";")
    if(!$PathFolders.Contains($fileName)){
        if($PathVariable.Trim().EndsWith(";")){
            $PathVariable = $PathVariable + $fileName
        }
        else {
            $PathVariable = $PathVariable + ";" + $fileName
        }
        [System.Environment]::SetEnvironmentVariable("Path",$PathVariable,"User")
    }
}

Add2Path -fileName $NNI_NODE_FOLDER
Add2Path -fileName "$NNI_YARN_FOLDER\bin"

# Refresh Path environment in this session
 $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
npm install -g node-pre-gyp
npm install -g react-scripts-ts-antd

# Building NNI Manager
cd src\nni_manager
cmd /c $NNI_YARN
cmd /c $NNI_YARN build
Copy-Item config -Destination .\dist\ -Recurse -Force
# Building WebUI
cd ..\webui
cmd /c $NNI_YARN
cmd /c $NNI_YARN build

## install-python-modules:
### Installing Python SDK
cd ..\sdk\pynni
(Get-Content setup.py).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content setup.py
cmd /c $PIP_INSTALL

## Installing nnictl
cd ..\..\..\tools 
(Get-Content setup.py).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content setup.py
cmd /c $PIP_INSTALL 

## install-node-modules
if(!(Test-Path $NNI_PKG_FOLDER)){
    New-Item $NNI_PKG_FOLDER -ItemType Directory
}
cd ..
Remove-Item $NNI_PKG_FOLDER -Recurse -Force
Copy-Item "src\nni_manager\dist" $NNI_PKG_FOLDER -Recurse
Copy-Item "src\nni_manager\package.json" $NNI_PKG_FOLDER
$PKG_JSON = $NNI_PKG_FOLDER + "\package.json"
(Get-Content $PKG_JSON).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content $PKG_JSON
cmd /c $NNI_YARN --prod --cwd $NNI_PKG_FOLDER
$NNI_PKG_FOLDER_STATIC = $NNI_PKG_FOLDER + "\static"
Copy-Item "src\webui\build" $NNI_PKG_FOLDER_STATIC -Recurse 
