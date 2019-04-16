param([bool]$test=$true)
python -m pip install --user --upgrade twine
if($test){
    python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*
}
else{
    python -m twine upload dist/*
}