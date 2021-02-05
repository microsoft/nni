自动补全 nnictl 命令
===================================

NNI的命令行工具 **nnictl** 支持自动补全，也就是说，您可以通过按 ``tab`` 键来完成 nnictl 命令。

例如当前命令是

.. code-block:: bash

   nnictl cre

按下 ``tab`` 键，它可以被自动补全成：

.. code-block:: bash

   nnictl create

目前，如果您通过 ``pip`` 安装 NNI ，默认情况下不会启用自动补全，并且它只在bash shell 的 Linux 上工作。 如果要启用此功能，请参阅以下步骤：

步骤 1. 下载 ``bash-completion``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd ~
   wget https://raw.githubusercontent.com/microsoft/nni/{nni-version}/tools/bash-completion

{nni-version} 应该填充 NNI 的版本，例如 ``master``\ , ``v1.9``。 你也可以 :githublink:`在这里 <tools/bash-completion>` 查看最新的 ``bash-completion`` 脚本。

步骤 2. 安装脚本
^^^^^^^^^^^^^^^^^^^^^^^^^^

如果您运行的是 root 帐户并希望为所有用户安装此脚本

.. code-block:: bash

   install -m644 ~/bash-completion /usr/share/bash-completion/completions/nnictl

如果您只是想自己安装这个脚本

.. code-block:: bash

   mkdir -p ~/.bash_completion.d
   install -m644 ~/bash-completion ~/.bash_completion.d/nnictl
   echo '[[ -f ~/.bash_completion.d/nnictl ]] && source ~/.bash_completion.d/nnictl' >> ~/.bash_completion

步骤 3. 重启终端
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

重新打开终端，您应该能够使用自动补全功能。 享受它吧！

步骤 4. 卸载
^^^^^^^^^^^^^^^^^

如果要卸载此功能，只需还原上述步骤中的更改。
