# Makefile 文件和安装配置

NNI 使用 GNU 来生成和安装。

`Makefile` 提供标准的目标 `生成`、`安装` 和 `卸载`, 以及不同设置的安装对象：

* `dev-easy-install`: 针对非专家用户，自动处理所有内容；
* `dev-easy-install`: 针对专家用户，自动处理所有内容；
* `install`: 针对 NNI 普通用户，通过复制文件来安装 NNI;
* `dev-install`: 针对 NNI 贡献者，通过创建 symlinks 而不是复制文件来安装 NNI;
* `pip-install`: 针对使用 `setup.py` 安装的情况;

下文会有更详细的介绍。

## 依赖项

NNI 依赖于 Node.js, Yarn, 和 pip 来生成，推荐安装 TypeScript。

NNI 需要 Node.js 以及运行所需要的所有库。 需要的 Node.js 库 (包括 TypeScript) 可以通过 Yarn 来安装， 需要的 Python 库可以通过 setuptools 或者 PIP 来安装。

NNI *用户*可以用 `make install-dependencies` 来安装 Node.js 和 Yarn。 Node.js 会被安装到 NNI 的安装目录，Yarn 会被安装到 `/tmp/nni-yarn`。 安装过程需要 wget。

NNI *开发人员*推荐手工安装 Node.js 和 Yarn。 可浏览相应的官方文档了解安装过程。

## 生成 NNI

当依赖项安装好后，运行 `make` 即可。

## 安装

### 目录结构

NNI 项目主要由两个 Node.js 模块 (`nni_manager`, `webui`) 以及两个 Python 包 (`nni`, `nnictl`) 所组成。

By default the Node.js modules are installed to `/usr/share/nni` for all users or installed to `~/.local/nni` for current user.

The Python packages are installed with setuptools and therefore the location depends on Python configuration. When install as non-priviledged user and virtualenv is not detected, `--user` flag will be used.

In addition, `nnictl` offers a bash completion scripts, which will be installed to `/usr/share/bash-completion/completions` or `~/.bash_completion.d`.

In some configuration, NNI will also install Node.js to `/usr/share/nni`.

All directories mentioned above are configurable. See next section for details.

### Configuration

The `Makefile` uses environment variables to override default settings.

Available variables are listed below:

| Name               | Description                                             | Default for normal user           | Default for root                                |
| ------------------ | ------------------------------------------------------- | --------------------------------- | ----------------------------------------------- |
| `BIN_PATH`         | Path for executables                                    | `~/.local/bin`                    | `/usr/bin`                                      |
| `INSTALL_PREFIX`   | Path for Node.js modules (a suffix `nni` will be added) | `~/.local`                        | `/usr/share`                                    |
| `BASH_COMP_SCRIPT` | Path of bash completion script                          | `~/.bash_completion.d/nnictl`     | `/usr/share/bash-completion/completions/nnictl` |
| `PIP_MODE`         | Arguments for `python3 setup.py install`                | `--user` if `VIRTUAL_ENV` not set | (empty)                                         |
| `NODE_PATH`        | Path to install Node.js runtime                         | `$INSTALL_PREFIX/nni/node`        | `$INSTALL_PREFIX/nni/node`                      |
| `YARN_PATH`        | Path to install Yarn                                    | `/tmp/nni-yarn`                   | `/tmp/nni-yarn`                                 |
| `NODE`             | Node.js command                                         | see source file                   | see source file                                 |
| `YARN`             | Yarn command                                            | see source file                   | see source file                                 |

Note that these variables will influence installation destination as well as generated `nnictl` and `nnimanager` scripts. If the path to copy files is different from where they will run (e.g. when creating a distro package), please generate `nnictl` and `nnimanager` manually.

### Targets

The workflow of each installation targets is listed below:

| Target             | Workflow                                                                   |
| ------------------ | -------------------------------------------------------------------------- |
| `easy-install`     | Install dependencies, build, install NNI, and edit `~/.bashrc`             |
| `dev-easy-install` | Install dependencies, build, install NNI as symlinks, and edit `~/.bashrc` |
| `install`          | Install Python packages, Node.js modules, NNI scripts, and examples        |
| `dev-install`      | Install Python and Node.js modules as symlinks, then install scripts       |
| `pip-install`      | Install dependencies, build, install NNI excluding Python packages         |

## TODO

* `clean` target
* `test` target
* `lint` target
* Test cases for each target
* Review variables