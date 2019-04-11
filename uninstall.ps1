
$NNI_DEPENDENCY_FOLDER = "C:\tmp\$env:USERNAME"

$WHICH_PYTHON = where.exe python;
if($WHICH_PYTHON[0].Length -eq 1){
    $NNI_PYTHON3 = $WHICH_PYTHON.SubString(0,$WHICH_PYTHON.Length-11)
}
else{
    $NNI_PYTHON3 = $WHICH_PYTHON[0].SubString(0,$WHICH_PYTHON[0].Length-11)
}

$PIP_UNINSTALL = "$NNI_PYTHON3\python -m pip uninstall -y "
$NNI_PKG_FOLDER = $NNI_PYTHON3 +"\nni"
$NNI_NODE_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-node"
$NNI_YARN_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-yarn"
 
# uninstall
Remove-Item $NNI_PKG_FOLDER -Recurse -Force
cmd /C $PIP_UNINSTALL "nni-sdk"
cmd /C $PIP_UNINSTALL "nni-tool"

# clean
Remove-Item "src/nni_manager/dist" -Recurse -Force
Remove-Item "src/nni_manager/node_modules" -Recurse -Force
Remove-Item "src/webui/build" -Recurse -Force
Remove-Item "src/webui/node_modules" -Recurse -Force
Remove-Item $NNI_YARN_FOLDER -Recurse -Force
Remove-Item $NNI_NODE_FOLDER -Recurse -Force

function Delete2Path {
    param ($fileName)
    $PathVariable = [System.Environment]::GetEnvironmentVariable("Path","User")
    $PathFolders = $PathVariable -split $fileName
    $PathVariable = $PathFolders -join ''
    [System.Environment]::SetEnvironmentVariable("Path",$PathVariable,"User")
}

$NNI_NODE_FOLDER = $NNI_NODE_FOLDER -split '\\' -join '\\'
$NNI_YARN_FOLDER = $NNI_YARN_FOLDER -split '\\' -join '\\'
Delete2Path -fileName $NNI_NODE_FOLDER
Delete2Path -fileName "$NNI_YARN_FOLDER\\bin"