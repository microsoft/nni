# cd docs
# python3 -m pip install --user -r requirements.txt
# sudo apt install pandoc
# cd en_US
rm -r build
sphinx-autobuild ./ build/html
# a=`netstat -tunlp|grep 8000`
# b=echo $a|awk -F " " '{print $6}'
# echo $b

