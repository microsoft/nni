# Makefile 文件和安装配置

NNI 使用 GNU Make 来生成和安装。

`Makefile` 提供标准的目标 `生成`、`安装` 和 `卸载`, 以及不同设置的安装对象：

* `easy-install`: 针对非专家用户，自动安装所有内容；
* `dev-easy-install`: 针对专家用户，自动安装所有内容；
* `install`: 针对 NNI 普通用户，通过复制文件来安装 NNI;
* `dev-install`: 针对 NNI 开发人员，通过创建符号链接而不是复制文件来安装 NNI;
* `pip-install`: 针对使用 `setup.py` 安装的情况;

下文会有更详细的介绍。

## 依赖项

NNI 依赖于 Node.js, Yarn, 和 PIP 来生成，推荐安装 TypeScript。

运行 NNI 需要 Node.js 以及依赖库。 Node.js 依赖库 (包括 TypeScript) 可以通过 Yarn 来安装， Python 依赖库可以通过 setuptools 或者 PIP 来安装。

NNI *普通用户*可以用 `make install-dependencies` 来安装 Node.js 和 Yarn。 Node.js 会被安装到 NNI 的安装目录，Yarn 会被安装到 `/tmp/nni-yarn`。 安装过程需要 wget。

NNI *开发人员*推荐手工安装 Node.js 和 Yarn。 可浏览相应的官方文档了解安装过程。

## 生成 NNI

当依赖项安装好后，运行 `make` 即可。

## 安装

### 目录层次

NNI 项目主要由两个 Node.js 模块 (`nni_manager`, `webui`) 以及两个 Python 包 (`nni`, `nnictl`) 所组成。

默认情况下，Node.js 模块为所有用户安装在 `/usr/share/nni` 目录下，也可以只为当前用户安装在 `~/.local/nni` 目录下。

Python 包使用 setuptools 安装，所以安装路径依赖于 Python 配置。 如果以非管理员身份安装 ，并且没有虚拟环境的时候，要加上 `--user` 参数。

此外，`nnictl` 提供一个自动完成脚本，会被安装在 `/usr/share/bash-completion/completions` 或 `~/.bash_completion.d` 目录下。

在某些配置情况下，NNI 也会将 Node.js 安装到 `/usr/share/nni` 目录下。

以上所有目录都是可配置的。 可参考下一章节。

### 配置

`Makefile` 中可以用环境变量来替换默认设置。

支持的变量如下：

| 名称                 | 说明                             | 普通用户下的默认值                          | root 下的默认值                                      |
| ------------------ | ------------------------------ | ---------------------------------- | ----------------------------------------------- |
| `BIN_PATH`         | 执行文件路径                         | `~/.local/bin`                     | `/usr/bin`                                      |
| `INSTALL_PREFIX`   | Node.js 模块的路径 (最后会加上 `nni`)    | `~/.local`                         | `/usr/share`                                    |
| `BASH_COMP_SCRIPT` | Bash 自动完成脚本的路径                 | `~/.bash_completion.d/nnictl`      | `/usr/share/bash-completion/completions/nnictl` |
| `PIP_MODE`         | `python3 setup.py install` 的参数 | 如果 `VIRTUAL_ENV` 没有设置，会加上 `--user` | (无)                                             |
| `NODE_PATH`        | Node.js 运行时的路径                 | `$INSTALL_PREFIX/nni/node`         | `$INSTALL_PREFIX/nni/node`                      |
| `YARN_PATH`        | Yarn 的安装路径                     | `/tmp/nni-yarn`                    | `/tmp/nni-yarn`                                 |
| `NODE`             | Node.js 命令                     | 参考源代码                              | 参考源代码                                           |
| `YARN`             | Yarn 命令                        | 参考源代码                              | 参考源代码                                           |

注意，这些变量不仅会影响安装路径，也会影响生成的 `nnictl` 脚本。 如果复制文件的路径和运行时的不一样（例如，创建发行版本包时），需要手工编辑 `nnictl` 和 `nnimanager`。

### 目标

安装目标的流程如下：

| 目标                 | 流程                                         |
| ------------------ | ------------------------------------------ |
| `easy-install`     | 安装依赖项，生成，安装 NNI，并编辑 `~/.bashrc`            |
| `dev-easy-install` | 安装依赖项，生成，将 NNI 作为符号链接来安装，并编辑 `~/.bashrc`   |
| `install`          | 安装 Python 包，Node.js 模块，NNI 脚本和示例           |
| `dev-install`      | 将 Python 和 Node.js 模块作为符号链接安装，然后安装 scripts |
| `pip-install`      | 安装依赖项，生成，安装 NNI，但不安装 Python 包              |

## TODO

* `clean` 目标
* `test` 目标
* `lint` 目标
* 每个目标的测试用例
* 评审变量