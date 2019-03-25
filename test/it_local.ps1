cmd /c python -m pip install scikit-learn==0.20.0 
cmd /c python -m pip install https://download.pytorch.org/whl/cu90/torch-0.4.1-cp36-cp36m-win_amd64.whl
cmd /c python -m pip install torchvision==0.2.1
cmd /c python -m pip install keras==2.1.6
cmd /c python -m pip install tensorflow-gpu==1.12.0
$swigUrl = "http://prdownloads.sourceforge.net/swig/swigwin-3.0.12.zip"
$NNI_DEPENDENCY_FOLDER = "\tmp\$env:USERNAME"
$NNI_SWIG_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-swig"
$NNI_SWIG_ZIP = $NNI_DEPENDENCY_FOLDER+"\nni-swig.zip"
$unzipSwigDir = "swigwin-3.0.12"

Write-Host "Downloading swig..."
(New-Object Net.WebClient).DownloadFile($swigUrl, $NNI_SWIG_ZIP)

if(Test-Path $NNI_SWIG_FOLDER){
    Remove-Item $NNI_SWIG_FOLDER -r -fo
}
Expand-Archive $NNI_SWIG_ZIP -DestinationPath $NNI_DEPENDENCY_FOLDER
Rename-Item "$NNI_DEPENDENCY_FOLDER\$unzipSwigDir" "nni-swig"

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

Add2Path -fileName $NNI_SWIG_FOLDER

cmd /c nnictl package install --name=SMAC


.\unittest.ps1


cmd /c python naive_test.py
cmd /c python tuner_test.py
cmd /c python config_test.py --ts local --local_gpu
cmd /c python metrics_test.py
