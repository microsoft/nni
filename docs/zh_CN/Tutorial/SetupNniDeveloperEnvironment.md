# 设置 NNI 开发环境

NNI 开发环境支持安装 Python 3 64 位的 Ubuntu 1604 （及以上）和 Windows 10。

## 安装

安装步骤与从源代码安装类似。 但是安装过程会链接到代码目录，以便代码改动能更方便的直接使用。

### 1. 克隆源代码

```bash
git clone https://github.com/Microsoft/nni.git
```

注意，如果要贡献代码，需要 Fork 自己的 NNI 代码库并克隆。

### 2. 从源代码安装

#### Ubuntu

```bash
make dev-easy-install
```

#### Windows

```bat
powershell -ExecutionPolicy Bypass -file install.ps1 -Development
```

### 3. 检查环境是否正确

可通过运行 Experiment，来检查环境。 例如，运行以下命令

```bash
nnictl create --config examples/trials/mnist-tfv1/config.yml
```

并打开 Web 界面查看

### 4. 重新加载改动

#### Python

无需操作，代码已连接到包的安装位置。

#### TypeScript

* 如改动了 `src/nni_manager`，在此目录下运行 `yarn watch` 可持续编译改动。 它将持续的监视并编译代码。 可能需要重启 `nnictl` 来重新加载 NNI 管理器。
* If `src/webui` is changed, run `yarn dev`, which will run a mock API server and a webpack dev server simultaneously. Use `EXPERIMENT` environment variable (e.g., `mnist-tfv1-running`) to specify the mock data being used. Built-in mock experiments are listed in `src/webui/mock`. An example of the full command is `EXPERIMENT=mnist-tfv1-running yarn dev`.
* If `src/nasui` is changed, run `yarn start` under the corresponding folder. The web UI will refresh automatically if code is changed. There is also a mock API server that is useful when developing. It can be launched via `node server.js`.

### 5. 提交拉取请求

所有改动都需要从自己 Fork 的代码库合并到 master 分支上。 拉取请求的描述必须有意义且有用。

我们会尽快审查更改。 审查通过后，我们会将代码合并到主分支。

有关更多贡献指南和编码风格，查看[贡献文档](Contributing.md)。