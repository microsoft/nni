import os
import torchvision.datasets.mnist as mnist

file_name = mnist.__file__
dummy_file_name = os.path.join(os.path.dirname(file_name), 'mnist_dummy.py')

with open(file_name, 'r') as fr:
    firstline = fr.readline()
    if firstline != 'from six.moves import urllib\n':
        with open(dummy_file_name, 'w') as fw:
            fw.writelines(['from six.moves import urllib\n',
                           'opener = urllib.request.build_opener()\n',
                           'opener.addheaders = [("User-agent", "Mozilla/5.0")]\n',
                           'urllib.request.install_opener(opener)\n\n'])
            fw.write(firstline)
            for line in fr:
                fw.write(line)

if os.path.exists(dummy_file_name):
    os.remove(file_name)
    os.rename(dummy_file_name, file_name)
