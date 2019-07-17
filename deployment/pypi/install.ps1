param([int]$version_os=64, [bool]$version_ts=$false)
[System.Net.ServicePointManager]::DefaultConnectionLimit = 100
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$CWD = $PWD

$OS_SPEC = "windows"
if($version_os -eq 64){
    $OS_VERSION = 'win64'
    $WHEEL_SPEC = 'win_amd64'
}
else{
    $OS_VERSION = 'win32'
    $WHEEL_SPEC = 'win32'
}

$TIME_STAMP = date -u "+%y%m%d%H%M"
$NNI_VERSION_VALUE = git describe --tags --abbrev=0

# To include time stamp in version value, run:
# make version_ts=true build

if($version_ts){
    $NNI_VERSION_VALUE = "$NNI_VERSION_VALUE.$TIME_STAMP"
}

$NNI_VERSION_TEMPLATE = "999.0.0-developing"

python -m pip install --upgrade setuptools wheel

$nodeUrl = "https://aka.ms/nni/nodejs-download/" + $OS_VERSION
$NNI_NODE_ZIP = "$CWD\node-$OS_SPEC.zip"
$NNI_NODE_FOLDER = "$CWD\node-$OS_SPEC"
$yarnUrl = "https://yarnpkg.com/latest.tar.gz"
$NNI_YARN_TARBALL = "$CWD\nni-yarn.tar.gz"
$NNI_YARN_FOLDER = "$CWD\nni-yarn"
$NNI_YARN = $NNI_YARN_FOLDER +"\bin\yarn"
$unzipYarnDir = "yarn-v*"
$nugetUrl = "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
$NNI_NUGET_FOLDER = "$CWD\nni-nuget"
$NNI_NUGET = $NNI_NUGET_FOLDER + "\nuget.exe"

(New-Object Net.WebClient).DownloadFile($nodeUrl, $NNI_NODE_ZIP)
(New-Object Net.WebClient).DownloadFile($yarnUrl, $NNI_YARN_TARBALL)
if (!(Test-Path $NNI_NUGET_FOLDER)) {
    New-Item $NNI_NUGET_FOLDER -ItemType Directory
}
(New-Object Net.WebClient).DownloadFile($nugetUrl, $NNI_NUGET)

$NNI_YARN_TARBALL = $NNI_YARN_TARBALL -split '\\' -join '\\'
$CWD = $CWD -split '\\' -join '\\'
$SCRIPT_PATH = $CWD + '\extract.py'
$SCRIPT =  "import tarfile",
       ("tar = tarfile.open(""{0}"")" -f $NNI_YARN_TARBALL),
       ("tar.extractall(""{0}"")" -f $CWD),
        "tar.close()"
[System.IO.File]::WriteAllLines($SCRIPT_PATH, $SCRIPT)

if(!(Test-Path $NNI_NODE_FOLDER)){
    New-Item $NNI_NODE_FOLDER -ItemType Directory
    cmd /c tar -xf $NNI_NODE_ZIP -C $NNI_NODE_FOLDER --strip-components 1
}

if(!(Test-Path $NNI_YARN_FOLDER)){
    cmd /c python $SCRIPT_PATH
    $unzipYarnDir = Get-ChildItem "$CWD\$unzipYarnDir"
    Rename-Item $unzipYarnDir "nni-yarn"
}

$env:PATH = $NNI_NODE_FOLDER+';'+$env:PATH
cd $CWD\..\..\src\nni_manager
yarn
yarn build
cd $CWD\..\..\src\webui
yarn
yarn build

# Building Aether Client
cd $CWD\..\..\src\nni_manager\training_service\aether\cslib
if (!(Get-Command msbuild | Test-Path)) {
    Write-Host "Please install msbuild first"
    exit
}
cmd /c $NNI_NUGET restore
cmd /c 'msbuild -Property:Configure=Release;OutputPath=..\..\..\dist\aether'

if(Test-Path $CWD\nni){
    Remove-Item $CWD\nni -Recurse -Force
}
Copy-Item $CWD\..\..\src\nni_manager\dist $CWD\nni -Recurse
Copy-Item $CWD\..\..\src\webui\build $CWD\nni\static -Recurse
Copy-Item $CWD\..\..\src\nni_manager\package.json $CWD\nni
(Get-Content $CWD\nni\package.json).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content $CWD\nni\package.json
cd $CWD\nni
cmd /c $NNI_YARN --prod
cd $CWD
(Get-Content setup.py).replace($NNI_VERSION_TEMPLATE, $NNI_VERSION_VALUE) | Set-Content setup.py
python setup.py bdist_wheel -p $WHEEL_SPEC
