
$NNI_DEPENDENCY_FOLDER = "\tmp\$env:USERNAME"
$NNI_PYTHON3 = "C:\Python3"
$NNI_PKG_FOLDER = $NNI_PYTHON3 +"\python\nni"
function FindPython{
    param()
    $val =  where.exe python;
    if($val -eq $null) {
        return "-1";
    }
    else {
        return $val;
    }
}
$WHICH_PYTHON = FindPython
if($WHICH_PYTHON -ne "-1"){
    $NNI_PYTHON3 = $WHICH_PYTHON.SubString(0,$WHICH_PYTHON.Length-11)
    $NNI_PKG_FOLDER = $NNI_PYTHON3 +"\nni"
}
$BIN_FOLDER = $NNI_PYTHON3+"\bin"
$NNI_NODE = $BIN_FOLDER+"\node" 
$LIB_FOLDER = $NNI_PYTHON3 + "\Lib\site-packages"
$NNI_LIB_FOLDER = $LIB_FOLDER + "\nni*"
$BASH_COMP_PREFIX = $env:HOMEPATH +"\.bash_completion.d"
$BASH_COMP_SCRIPT = $BASH_COMP_PREFIX +"\nnictl"
$NNI_NODE_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-node"
$NNI_YARN_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-yarn"
 
# uninstall
Remove-Item $NNI_PKG_FOLDER -r -fo
Remove-Item $NNI_LIB_FOLDER -r -fo
Remove-Item "$BIN_FOLDER/node" -r -fo
Remove-Item $BASH_COMP_SCRIPT -r -fo

# clean
Remove-Item "src/nni_manager/dist" -r -fo
Remove-Item "src/nni_manager/node_modules" -r -fo
Remove-Item "src/sdk/pynni/build" -r -fo
Remove-Item "src/webui/build" -r -fo
Remove-Item "src/webui/node_modules" -r -fo
Remove-Item $NNI_YARN_FOLDER -r -fo
Remove-Item $NNI_NODE_FOLDER -r -fo 

function Delete2Path {
    param ($fileName)
    $PathVariable = [System.Environment]::GetEnvironmentVariable("Path","Machine")
    $PathFolders = $PathVariable -split $fileName
    $PathVariable = $PathFolders -join ''
    [System.Environment]::SetEnvironmentVariable("Path",$PathVariable,"Machine")
}

$NNI_NODE = $NNI_NODE -split '\\' -join '\\'
$NNI_YARN_FOLDER = $NNI_YARN_FOLDER -split '\\' -join '\\'
Delete2Path -fileName $NNI_NODE
Delete2Path -fileName "$NNI_YARN_FOLDER\\bin"
