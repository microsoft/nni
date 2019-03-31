param([bool]$version_ts=$false)
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
$nodeUrl = "https://nodejs.org/dist/v10.15.1/node-v10.15.1-win-x64.zip"
$version = "10.15.1"
$unzipNodeDir = "node-v$version-win-x64"
(New-Object Net.WebClient).DownloadFile($nodeUrl, "$CWD\node-$OS_SPEC-x64.zip")
if(Test-Path "$CWD\node-$OS_SPEC-x64"){
    Remove-Item "$CWD\node-$OS_SPEC-x64" -r -fo
}
Expand-Archive "$CWD\node-$OS_SPEC-x64.zip" -DestinationPath $CWD
Rename-Item "$CWD\$unzipNodeDir" "node-$OS_SPEC-x64"

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