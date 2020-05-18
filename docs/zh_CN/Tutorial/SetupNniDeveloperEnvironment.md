# 设置 NNI 开发环境

NNI 开发环境支持安装 Python 3 64 位的 Ubuntu 1604 （及以上）和 Windows 10。

## 安装

安装步骤与从源代码安装类似。 但是安装过程会链接到代码目录，以便代码改动能更方便的直接使用。

### 1. 克隆源代码

```bash
git clone https://github.com/Microsoft/nni.git
```

Note, if you want to contribute code back, it needs to fork your own NNI repo, and clone from there.

### 2. Install from source code

#### Ubuntu

```bash
make dev-easy-install
```

#### Windows

```bat
powershell -ExecutionPolicy Bypass -file install.ps1 -Development
```

### 3. Check if the environment is ready

Now, you can try to start an experiment to check if your environment is ready. For example, run the command

```bash
nnictl create --config examples/trials/mnist-tfv1/config.yml
```

And open WebUI to check if everything is OK

### 4. Reload changes

#### Python

Nothing to do, the code is already linked to package folders.

#### TypeScript

* If `src/nni_manager` is changed, run `yarn watch` under this folder. It will watch and build code continually. The `nnictl` need to be restarted to reload NNI manager.
* 如果改动了 `src/webui` 或 `src/nasui`，在相应目录下运行 `yarn start`。 Web 界面会在代码修改后自动刷新。

### 5. Submit Pull Request

All changes are merged to master branch from your forked repo. The description of Pull Request must be meaningful, and useful.

We will review the changes as soon as possible. Once it passes review, we will merge it to master branch.

For more contribution guidelines and coding styles, you can refer to the [contributing document](Contributing.md).