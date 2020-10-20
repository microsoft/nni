$CWD = $PWD
$OS_SPEC = "windows"
Remove-Item $CWD\build -Recurse -Force
Remove-Item $CWD\dist -Recurse -Force
Remove-Item $CWD\nni -Recurse -Force
Remove-Item $CWD\nni.egg-info -Recurse -Force
Remove-Item $CWD\node-$OS_SPEC -Recurse -Force