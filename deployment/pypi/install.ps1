param([bool]$version_ts=$false)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$CWD = $PWD

$OS_SPEC = "windows"
$WHEEL_SPEC = "win_amd64"

$TIME_STAMP = date -u "+%y%m%d%H%M"
$NNI_VERSION_VALUE = git describe --tags --abbrev=0

# To include time stamp in version value, run:
# make version_ts=true build

if($version_ts){
    $NNI_VERSION_VALUE = "$NNI_VERSION_VALUE.$TIME_STAMP"
}

$NNI_VERSION_TEMPLATE = "999.0.0-developing"

python -m pip install --user --upgrade setuptools wheel

$nodeUrl = "https://aka.ms/nni/nodejs-download/win64"
$NNI_NODE_ZIP = "$CWD\node-$OS_SPEC-x64.zip"
$NNI_NODE_FOLDER = "$CWD\node-$OS_SPEC-x64"
(New-Object Net.WebClient).DownloadFile($nodeUrl, $NNI_NODE_ZIP)
if(Test-Path $NNI_NODE_FOLDER){
    Remove-Item $NNI_NODE_FOLDER -Recurse -Force
}
New-Item $NNI_NODE_FOLDER -ItemType Directory
cmd /c tar -xf $NNI_NODE_ZIP -C $NNI_NODE_FOLDER --strip-components 1

cd $CWD\..\..\src\nni_manager
yarn
yarn build
cd $CWD\..\..\src\webui 
yarn
yarn build
if(Test-Path $CWD\nni){
    Remove-Item $CWD\nni -r -fo
}
Copy-Item $CWD\..\..\src\nni_manager\dist $CWD\nni -Recurse
Copy-Item $CWD\..\..\src\webui\build $CWD\nni\static -Recurse
Copy-Item $CWD\..\..\src\nni_manager\package.json $CWD\nni
(Get-Content $CWD\nni\package.json).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content $CWD\nni\package.json
cd $CWD\nni
yarn --prod
cd $CWD
(Get-Content setup.py).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content setup.py
python setup.py bdist_wheel -p $WHEEL_SPEC