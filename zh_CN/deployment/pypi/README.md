# Python Package Index (PyPI) for NNI

## 1.Description

This is the PyPI build and upload tool for NNI project.

## 2.Prepare environment

Before build and upload NNI package, make sure the below OS and tools are available.

    Ubuntu 16.04 LTS
    make
    wget
    Python >= 3.5
    Pip
    Node.js
    Yarn
    

## 2.How to build

```bash
make
```

## 3.How to upload

### upload for testing

```bash
TWINE_REPOSITORY_URL=https://test.pypi.org/legacy/ make upload
```

You may need to input the account and password of https://test.pypi.org during this process.

### upload for release

```bash
make upload
```

You may need to input the account and password of https://pypi.org during this process.