# 用于 NNI 的 python 包索引 (pypi)

这是用于 NNI 项目的 PyPI 生成和上传的工具。

## **For Linux**

* **Prepare environment**
    
    Before build and upload NNI package, make sure the below OS and tools are available.
    
        Ubuntu 16.04 LTS
        make
        wget
        Python >= 3.5
        Pip
        Node.js
        Yarn
        

* **How to build**
    
    ```bash
    make
    ```

* **How to upload**
    
    **upload for testing**
    
    ```bash
    TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/ make upload
    ```
    
    You may need to input the account and password of https://test.pypi.org during this process.
    
    **upload for release**
    
    ```bash
    make upload
    ```
    
    You may need to input the account and password of https://pypi.org during this process.

## **For Windows**

* **Prepare environment**
    
    Before build and upload NNI package, make sure the below OS and tools are available.
    
        Windows 10
        powershell
        Python >= 3.5
        Pip
        Node.js
        Yarn
        tar
        

* **How to build**
    
    ```bash
    powershell ./install.ps1
    ```

* **How to upload**
    
    **upload for testing**
    
    ```bash
    powershell ./upload.ps1
    ```
    
    You may need to input the account and password of https://test.pypi.org during this process.
    
    **upload for release**
    
    ```bash
    powershell ./upload.ps1 -test $False
    ```
    
    You may need to input the account and password of https://pypi.org during this process.