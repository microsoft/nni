$NNI_DEPENDENCY_FOLDER = [System.IO.Path]::GetTempPath()+$env:USERNAME

$env:PYTHONIOENCODING = "UTF-8"
if($env:VIRTUAL_ENV){
    $NNI_PYTHON3 = $env:VIRTUAL_ENV + "\Scripts"
    $NNI_PKG_FOLDER = $env:VIRTUAL_ENV + "\nni"
    cmd /c if exist "$NNI_PYTHON3\node.exe" del "$NNI_PYTHON3\node.exe"
}
else{
    $NNI_PYTHON3 = $(python -c 'import site; from pathlib import Path; print(Path(site.getsitepackages()[0]))')
    $NNI_PKG_FOLDER = $NNI_PYTHON3 + "\nni"
    cmd /c if exist "$NNI_PYTHON3\Scripts\node.exe" del "$NNI_PYTHON3\Scripts\node.exe"
}

$PIP_UNINSTALL = """$NNI_PYTHON3\python"" -m pip uninstall -y "
$NNI_NODE_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-node"
$NNI_YARN_FOLDER = $NNI_DEPENDENCY_FOLDER+"\nni-yarn"

# uninstall
cmd /c if exist "$NNI_PKG_FOLDER" rmdir /s /q $NNI_PKG_FOLDER
cmd /c $PIP_UNINSTALL "nni"

# clean up
cmd /c if exist "build" rmdir /s /q "build"
cmd /c if exist "src\nni_manager\dist" rmdir /s /q "src\nni_manager\dist"
cmd /c if exist "src\nni_manager\node_modules" rmdir /s /q  "src\nni_manager\node_modules"
cmd /c if exist "src\webui\build" rmdir /s /q  "src\webui\build"
cmd /c if exist "src\webui\node_modules" rmdir /s /q  "src\webui\node_modules"
cmd /c if exist "src\nasui\build" rmdir /s /q  "src\nasui\build"
cmd /c if exist "src\nasui\node_modules" rmdir /s /q  "src\nasui\node_modules"
cmd /c if exist "$NNI_YARN_FOLDER" rmdir /s /q  $NNI_YARN_FOLDER
cmd /c if exist "$NNI_NODE_FOLDER" rmdir /s /q  $NNI_NODE_FOLDER
