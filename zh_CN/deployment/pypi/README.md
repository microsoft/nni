# 用于 NNI 的 python 包索引 (pypi)

## 1. 说明

这是用于 NNI 项目的 PyPI 生成和上传的工具。

## 2.准备环境

在生成和上传 NNI 包之前，确保使用了下列环境。

    Ubuntu 16.04 LTS
    make
    wget
    Python >= 3.5
    Pip
    Node.js
    Yarn
    

## 2.如何生成

```bash
make
```

## 3.如何上传

### 上传测试包

```bash
TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/ make upload
```

上传过程中，可能需要输入 https://test.pypi.org 的用户和密码。

### 上传发布包

```bash
make upload
```

上传过程中，可能需要输入 https://pypi.org 的用户和密码。