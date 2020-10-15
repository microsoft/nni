# 用于 NNI 的 python 包索引 (pypi)

这是用于 NNI 项目的 PyPI 生成和上传的工具。

## **Linux**

* **准备环境**
    
    在生成和上传 NNI 包之前，确保使用了下列环境。
    
        Ubuntu 16.04 LTS
        make
        wget
        Python >= 3.6
        Pip
        Node.js
        Yarn
        

* **如何生成**
    
    ```bash
    make
    ```

* **如何上传**
    
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

* **准备环境**
    
    在生成和上传 NNI 包之前，确保使用了下列环境。
    
        Windows 10
        powershell
        Python >= 3.6
        Pip
        Yarn
        

* **如何生成**
    
    参数 `version_os` 用来选择使用 64 位还是 32 位 Windows 来生成。
    
    ```bash
    powershell ./install.ps1 -version_os [64/32]
    ```

* **如何上传**
    
    **upload for testing**
    
    ```bash
    powershell ./upload.ps1
    ```
    
    上传过程中，可能需要输入 https://test.pypi.org 的用户和密码。
    
    **上传发布包**
    
    ```bash
    powershell ./upload.ps1 -test $False
    ```
    
    上传过程中，可能需要输入 https://pypi.org 的用户和密码。