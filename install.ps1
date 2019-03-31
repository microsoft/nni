# activate / desactivate any install
$install_node = $TRUE
$install_yarn = $TRUE

### CONFIGURATION
$PIP_INSTALL = "python -m pip install ."

# nodejs
$version = "10.15.1"
$nodeUrl = "https://nodejs.org/dist/v10.15.1/node-v10.15.1-win-x64.zip"
$yarnUrl = "https://yarnpkg.com/latest.tar.gz"
$unzipNodeDir = "node-v$version-win-x64"

$NNI_DEPENDENCY_FOLDER = "\tmp\$env:USERNAME"

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
$NNI_PYTHON3 = $WHICH_PYTHON.SubString(0,$WHICH_PYTHON.Length-11)
$NNI_PKG_FOLDER = $NNI_PYTHON3 +"\nni"

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