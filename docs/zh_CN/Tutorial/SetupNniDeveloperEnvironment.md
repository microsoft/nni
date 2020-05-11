**设置 NNI 开发环境**

===

## 调试 NNI 源代码的最佳实践

要调试 NNI 源代码，需要 Ubuntu 16.04 或更高版本系统的开发环境，并需要安装 Python 3 以及 pip 3，然后遵循以下步骤。

### 1. 克隆源代码

运行命令

    git clone https://github.com/Microsoft/nni.git
    

来克隆源代码

### 2. 准备调试环境并安装依赖项

将目录切换到源码目录，然后运行命令

    make install-dependencies
    

来安装环境的依赖项工具

### 3. 生成源代码

运行命令

    make build
    

来生成源代码

### 4. 将 NNI 安装到开发环境中

运行命令

    make dev-install
    

来安装分发内容到开发环境，并创建 cli 脚本

### 5. 检查环境是否正确

Trial 启动 Experiment 来检查环境。 例如，运行命令

    nnictl create --config ~/nni/examples/trials/mnist-tfv1/config.yml
    

并打开网页界面查看

### 6. 重新部署

代码更改后，可能需要重新部署。 这取决于更改了哪种代码。

#### Python

不需要重新部署，但可能需要重新启动 nnictl。

#### TypeScript

* 如果要更改 `src/nni_manager`，运行 `yarn watch` 可持续编译改动。 它将实时重建代码。 可能需要重新启动 nnictl 来重新加载 NNI 管理器。
* 如果改动了 `src/webui` 或 `src/nasui`，在相应目录下运行 `yarn start`。 Web 界面会在代码修改后自动刷新。

* * *

最后，希望一切顺利。 参考[贡献](Contributing.md)文档，来了解更多创建拉取请求或问题的指南。