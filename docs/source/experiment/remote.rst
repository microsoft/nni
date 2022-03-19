Remote Training Service
=======================

TBD

Temporarily placed here:

.. _nniignore:

.. Note:: If you are planning to use remote machines or clusters as your training service, to avoid too much pressure on network, NNI limits the number of files to 2000 and total size to 300MB. If your codeDir contains too many files, you can choose which files and subfolders should be excluded by adding a ``.nniignore`` file that works like a ``.gitignore`` file. For more details on how to write this file, see the `git documentation <https://git-scm.com/docs/gitignore#_pattern_format>`__.

*Example:* :githublink:`config_detailed.yml <examples/trials/mnist-pytorch/config_detailed.yml>` and :githublink:`.nniignore <examples/trials/mnist-pytorch/.nniignore>`
