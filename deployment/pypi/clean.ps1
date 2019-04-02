$CWD = $PWD
$OS_SPEC = "windows"
Remove-Item $CWD\build -r -fo
Remove-Item $CWD\dist -r -fo
Remove-Item $CWD\nni -r -fo
# TO DO
Remove-Item $CWD\nni.egg-info -r -fo
Remove-Item $CWD\node-$OS_SPEC-x64 -r -fo