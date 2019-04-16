# 用于 NNI 的 python 包索引 (pypi)

这是用于 NNI 项目 PyPI 生成和上传的工具。

## **Linux**

* __准备环境__

  在生成和上传 NNI 包之前，确保使用了下列环境。
  ```
  Ubuntu 16.04 LTS
  make
  wget
  Python >= 3.5
  Pip
  Node.js
  Yarn
  ```

* __如何生成__

  ```bash
  make
  ```

* __如何上传__

  **上传测试包**
  ```bash
  TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/ make upload
  ```
  上传过程中，可能需要输入 https://test.pypi.org 的用户和密码。

  **上传发布包**
  ```bash
  make upload
  ```
  上传过程中，可能需要输入 https://pypi.org 的用户和密码。

## **Windows**

* __准备环境__

  在生成和上传 NNI 包之前，确保使用了下列环境。
  ```
  Windows 10
  powershell
  Python >= 3.5
  Pip
  Node.js
  Yarn
  ```

* __如何生成__

  ```bash
  powershell ./install.ps1
  ```

* __如何上传__

  **上传测试包**
  ```bash
  powershell ./upload.ps1
  ```
  上传过程中，可能需要输入 https://test.pypi.org 的用户和密码。

  **上传发布包**
  ```bash
  powershell ./upload.ps1 -test $False
  ```
  上传过程中，可能需要输入 https://pypi.org 的用户和密码。