
$NNI_DEPENDENCY_FOLDER = "C:\tmp\$env:USERNAME"

set PYTHONIOENCODING = UTF-8
if($env:VIRTUAL_ENV){
    $NNI_PYTHON3 = $env:VIRTUAL_ENV + "\Scripts"
    $NNI_PKG_FOLDER = $env:VIRTUAL_ENV + "\nni"
}
else{
    $NNI_PYTHON3 = $(python -c 'import site; from pathlib import Path; print(Path(site.getsitepackages()[0]))')
    $NNI_PKG_FOLDER = $NNI_PYTHON3 + "\nni"
}

$PIP_UNINSTALL = """$NNI_PYTHON3\python"" -m pip uninstall -y "
Remove-Item "$NNI_PYTHON3\Scripts\node.exe" -Force
$NNI_NODE_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-node"
$NNI_YARN_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-yarn"
 
# uninstall
Remove-Item $NNI_PKG_FOLDER -Recurse -Force
cmd /C $PIP_UNINSTALL "nni"

# clean
Remove-Item "src/nni_manager/dist" -Recurse -Force
Remove-Item "src/nni_manager/node_modules" -Recurse -Force
Remove-Item "src/webui/build" -Recurse -Force
Remove-Item "src/webui/node_modules" -Recurse -Force
Remove-Item $NNI_YARN_FOLDER -Recurse -Force
Remove-Item $NNI_NODE_FOLDER -Recurse -Force