# activate / desactivate any install
$install_node = $TRUE
$install_yarn = $TRUE
$install_py = $FALSE
$install_pip = $FALSE

### CONFIGURATION
$PIP_INSTALL = "python -m pip install ."

# nodejs
$version = "10.15.1"
$pyVersion ="36"
$nodeUrl = "https://nodejs.org/dist/v10.15.1/node-v10.15.1-win-x64.zip"
$yarnUrl = "https://yarnpkg.com/latest.tar.gz"
$pyUrl= "https://www.python.org/ftp/python/3.6.4/python-3.6.4-embed-amd64.zip"
$pipUrl = "https://bootstrap.pypa.io/get-pip.py"
$unzipNodeDir = "node-v$version-win-x64"
$unzipPythonDir = "python-3.6.4-embed-amd64"

$NNI_DEPENDENCY_FOLDER = "\tmp\$env:USERNAME"
$NNI_PYTHON3 = "C:\Python3"
$NNI_PKG_FOLDER = $NNI_PYTHON3 +"\python\nni"

$WHICH_PYTHON = where.exe python
if($WHICH_PYTHON -eq $null){
    $install_py = $TRUE
    $NNI_PYTHON_FOLDER = $NNI_PYTHON3 +"\python"
}
else {
    $NNI_PYTHON3 = $WHICH_PYTHON.SubString(0,$WHICH_PYTHON.Length-11)
    $NNI_PYTHON_FOLDER = $NNI_PYTHON3
    $NNI_PKG_FOLDER = $NNI_PYTHON3 +"\nni"
}

$WHICH_PIP = where.exe pip
if($WHICH_PIP -eq $null){
    $install_pip = $TRUE
}

$NNI_PYTHON3_ZIP = $NNI_PYTHON3 +"\python.zip"
$GET_PIP = $NNI_PYTHON3 +"\get-pip.py"
$NNI_PIP_FOLDER = $NNI_PYTHON_FOLDER+"\Scripts"
$BASH_COMP_PREFIX = $env:HOMEPATH +"\.bash_completion.d"
$BASH_COMP_SCRIPT = $BASH_COMP_PREFIX +"\nnictl"
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

if ($install_node) {

    ### nodejs install
    if(Test-Path $NNI_NODE_FOLDER){
        Remove-Item $NNI_NODE_FOLDER -r -fo
    }
    Expand-Archive $NNI_NODE_ZIP -DestinationPath $NNI_DEPENDENCY_FOLDER
    Rename-Item "$NNI_DEPENDENCY_FOLDER\$unzipNodeDir" "nni-node"
     
    ### yarn install
    if(Test-Path $NNI_YARN_FOLDER){
        Remove-Item $NNI_YARN_FOLDER -r -fo
    }
    New-Item $NNI_YARN_FOLDER -ItemType Directory
	cmd /c tar -xf $NNI_YARN_TARBALL -C $NNI_YARN_FOLDER --strip-components 1
    
}

if($install_py){
    if(!(Test-Path $NNI_PYTHON_FOLDER)){
        New-Item $NNI_PYTHON_FOLDER -ItemType Directory
    }
    Write-Host "Downloading Python3..."
    (New-Object Net.WebClient).DownloadFile($pyUrl, $NNI_PYTHON3_ZIP)
    Expand-Archive $NNI_PYTHON3_ZIP -DestinationPath $NNI_PYTHON_FOLDER
    # fix read zip error
    $PYTHON3_INNER_ZIP = "$NNI_PYTHON_FOLDER\python$pyVersion"+".zip"
    Expand-Archive $PYTHON3_INNER_ZIP -DestinationPath $PYTHON3_INNER_ZIP.Split('.')[0]
    $Rename_INNER_ZIP = "$NNI_PYTHON_FOLDER\python$pyVersion"+".zipp"
    Rename-Item $PYTHON3_INNER_ZIP $Rename_INNER_ZIP
    Rename-Item $PYTHON3_INNER_ZIP.Split('.')[0] $PYTHON3_INNER_ZIP
    # fix import local file error
    $deleteFile = $NNI_PYTHON_FOLDER + "\python36._pth"
    if(Test-Path $deleteFile){
        Remove-Item $deleteFile -r -fo
    }
}

if($install_pip){
    Write-Host "Downloading pip..."
    (New-Object Net.WebClient).DownloadFile($pipUrl, $GET_PIP)
    cmd /c "$NNI_PYTHON_FOLDER\python $GET_PIP"
}

### add to PATH
function Add2Path {
    param ($fileName)
    $PathVariable = [System.Environment]::GetEnvironmentVariable("Path","Machine")
    $PathFolders = $PathVariable.Split(";")
    if(!$PathFolders.Contains($fileName)){
        if($PathVariable.Trim().EndsWith(";")){
            $PathVariable = $PathVariable + $fileName
        }
        else {
            $PathVariable = $PathVariable + ";" + $fileName
        }
        [System.Environment]::SetEnvironmentVariable("Path",$PathVariable,"Machine")
    }
}

Add2Path -fileName $NNI_NODE_FOLDER
Add2Path -fileName "$NNI_YARN_FOLDER\bin"
if($install_py){
    Add2Path -fileName $NNI_PYTHON_FOLDER
}
if($install_pip){
    Add2Path -fileName $NNI_PIP_FOLDER
}

# Refresh Path environment in this session
foreach($level in "Machine","User") {
    [Environment]::GetEnvironmentVariables($level).GetEnumerator() | % {
       # For Path variables, append the new values, if they're not already in there
       if($_.Name -match 'Path$') { 
          $_.Value = ($((Get-Content "Env:$($_.Name)") + ";$($_.Value)") -split ';' | Select -unique) -join ';'
       }
       $_
    } | Set-Content -Path { "Env:$($_.Name)" }
 }

# Building NNI Manager
cd src\nni_manager
cmd /c $NNI_YARN
cmd /c $NNI_YARN build
Copy-Item config -Destination .\dist\ -Recurse  
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
Remove-Item $NNI_PKG_FOLDER -r -fo
Copy-Item "src\nni_manager\dist" $NNI_PKG_FOLDER -Recurse
Copy-Item "src\nni_manager\package.json" $NNI_PKG_FOLDER
$PKG_JSON = $NNI_PKG_FOLDER+"\package.json"
(Get-Content $PKG_JSON).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content $PKG_JSON
cmd /c $NNI_YARN --prod --cwd $NNI_PKG_FOLDER
$NNI_PKG_FOLDER_STATIC = $NNI_PKG_FOLDER+"\static"
Copy-Item "src\webui\build" $NNI_PKG_FOLDER_STATIC -Recurse 
if(!(Test-Path $BASH_COMP_PREFIX)){
    New-Item $BASH_COMP_PREFIX -ItemType Directory 
}
Copy-Item tools/bash-completion $BASH_COMP_SCRIPT
