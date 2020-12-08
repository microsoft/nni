# Python Package Index (PyPI) for NNI

This is the PyPI build and upload tool for NNI project.

## **For Linux**

* __Prepare environment__

  Before build and upload NNI package, make sure the below OS and tools are available.
  ```
  Ubuntu 16.04 LTS
  make
  wget
  Python >= 3.6
  Pip
  Node.js
  Yarn
  ```

* __How to build__

  ```bash
  make
  ```

* __How to upload__

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

* __Prepare environment__

  Before build and upload NNI package, make sure the below OS and tools are available.
  ```
  Windows 10
  powershell
  Python >= 3.6
  Pip
  Yarn
  ```

* __How to build__
 
  parameter `version_os` is used to build for Windows 64-bit or 32-bit. 
  ```bash
  powershell ./install.ps1 -version_os [64/32]
  ```

* __How to upload__

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
