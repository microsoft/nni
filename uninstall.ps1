
$NNI_DEPENDENCY_FOLDER = "\tmp\$env:USERNAME"

$WHICH_PYTHON = where.exe python;
$NNI_PYTHON3 = $WHICH_PYTHON.SubString(0,$WHICH_PYTHON.Length-11)
$NNI_PKG_FOLDER = $NNI_PYTHON3 +"\nni"
$LIB_FOLDER = $NNI_PYTHON3 + "\Lib\site-packages"
$NNI_LIB_FOLDER = $LIB_FOLDER + "\nni*"
$NNI_NODE_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-node"
$NNI_YARN_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-yarn"
 
# uninstall
Remove-Item $NNI_PKG_FOLDER -Recurse -Force
Remove-Item $NNI_LIB_FOLDER -Recurse -Force

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