param([int]$version_os, [bool]$version_ts=$false)
[System.Net.ServicePointManager]::DefaultConnectionLimit = 100
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$CWD = $PWD

$OS_SPEC = "windows"
if($version_os -eq 64){
    $OS_VERSION = 'win64'
    $WHEEL_SPEC = 'win_amd64'
    $NODE_URL = 'https://nodejs.org/download/release/v10.22.1/node-v10.22.1-win-x64.zip'
}
else{
    $OS_VERSION = 'win32'
    $WHEEL_SPEC = 'win32'
    $NODE_URL = 'https://nodejs.org/download/release/v10.22.1/node-v10.22.1-win-x86.zip'
}

$TIME_STAMP = date -u "+%y%m%d%H%M"
$NNI_VERSION_VALUE = git describe --tags --abbrev=0
$NNI_VERSION_VALUE = $NNI_VERSION_VALUE.substring(1)

# To include time stamp in version value, run:
# make version_ts=true build

if($version_ts){
    $NNI_VERSION_VALUE = "$NNI_VERSION_VALUE.$TIME_STAMP"
}

$NNI_VERSION_TEMPLATE = "999.0.0-developing"

python -m pip install --upgrade setuptools wheel

$NNI_NODE_ZIP = "$CWD\node-$OS_SPEC.zip"
$NNI_NODE_FOLDER = "$CWD\node-$OS_SPEC"
$unzipNodeDir = "node-v*"
(New-Object Net.WebClient).DownloadFile($NODE_URL, $NNI_NODE_ZIP)
if(Test-Path $NNI_NODE_FOLDER){
    Remove-Item $NNI_NODE_FOLDER -Recurse -Force
}
Expand-Archive $NNI_NODE_ZIP -DestinationPath $CWD
$unzipNodeDir = Get-ChildItem "$CWD\$unzipNodeDir"
Rename-Item $unzipNodeDir $NNI_NODE_FOLDER

$env:PATH = $NNI_NODE_FOLDER+';'+$env:PATH
cd $CWD\..\..\src\nni_manager
yarn
yarn build
Copy-Item config -Destination .\dist\ -Recurse -Force
cd $CWD\..\..\src\webui
yarn
yarn build
cd $CWD\..\..\src\nasui
yarn
yarn build
if(Test-Path $CWD\nni){
    Remove-Item $CWD\nni -r -fo
}
Copy-Item $CWD\..\..\src\nni_manager\dist $CWD\nni -Recurse
Copy-Item $CWD\..\..\src\webui\build $CWD\nni\static -Recurse
Copy-Item $CWD\..\..\src\nasui\build $CWD\nni\nasui -Recurse
Copy-Item $CWD\..\..\src\nasui\server.js $CWD\nni\nasui -Recurse
Copy-Item $CWD\..\..\src\nni_manager\package.json $CWD\nni
(Get-Content $CWD\nni\package.json).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content $CWD\nni\package.json
cd $CWD\nni
yarn --prod
cd $CWD\..\..\src\sdk\pynni\nni
(Get-Content __init__.py).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content __init__.py
cd $CWD
(Get-Content setup.py).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content setup.py
python setup.py bdist_wheel -p $WHEEL_SPEC
