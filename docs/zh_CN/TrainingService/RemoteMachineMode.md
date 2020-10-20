# 在远程计算机上运行 Experiment

NNI 可以通过 SSH 在多个远程计算机上运行同一个 Experiment，称为 `remote` 模式。 这就像一个轻量级的训练平台。 在此模式下，可以从计算机启动 NNI，并将 Trial 并行调度到远程计算机。

远程计算机操作系统支持 `Linux`, `Windows 10`, 和 `Windows Server 2019`。

## 必需组件

* 确保远程计算机的默认环境符合 Trial 代码的需求。 如果默认环境不符合要求，可以将设置脚本添加到 NNI 配置的 `command` 字段。

* 确保远程计算机能被运行 `nnictl` 命令的计算机通过 SSH 访问。 同时支持 SSH 的密码和密钥验证方法。 有关高级用法，参考[配置](../Tutorial/ExperimentConfig.md)的 machineList 部分。

* 确保每台计算机上的 NNI 版本一致。

* 如果要同时使用远程 Linux 和 Windows，请确保 Trial 的命令与远程操作系统兼容。 例如，Python 3.x 的执行文件在 Linux 下是 `python3`，在 Windows 下是 `python`。

### Linux

* 根据[安装说明](../Tutorial/InstallationLinux.md)，在远程计算机上安装 NNI。

### Windows

* 根据[安装说明](../Tutorial/InstallationWin.md)，在远程计算机上安装 NNI。

* 安装并启动 `OpenSSH Server`。
    
    1. 打开 Windows 中的`设置`应用。
    
    2. 点击`应用程序`，然后点击`可选功能`。
    
    3. 点击`添加功能`，搜索并选择 `OpenSSH Server`，然后点击`安装`。
    
    4. 安装后，运行下列命令来启动服务并设为自动启动。
    
    ```bat
    sc config sshd start=auto
    net start sshd
    ```

* 确保远程账户为管理员权限，以便可以停止运行中的 Trial。

* 确保除了默认消息外，没有别的欢迎消息，否则会导致 NodeJS 中的 ssh2 出错。 例如，如果在 Azure 中使用了 Data Science VM，需要删除 `C:\dsvm\tools\setup\welcome.bat` 中的 echo 命令。
    
    打开新命令窗口，如果输入如下，则表示正常。
    
    ```text
    Microsoft Windows [Version 10.0.17763.1192]
    (c) 2018 Microsoft Corporation. All rights reserved.
    
    (py37_default) C:\Users\AzureUser>
    ```

## 运行 Experiment

例如，有三台机器，可使用用户名和密码登录。

| IP       | 用户名 | 密码     |
| -------- | --- | ------ |
| 10.1.1.1 | bob | bob123 |
| 10.1.1.2 | bob | bob123 |
| 10.1.1.3 | bob | bob123 |

在这三台计算机或另一台能访问这些计算机的环境中安装并运行 NNI。

以 `examples/trials/mnist-annotation` 为例。 示例文件 `examples/trials/mnist-annotation/config_remote.yml` 的内容如下：

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: remote
# 搜索空间文件
searchSpacePath: search_space.json
# 可选项: true, false
useAnnotation: true
tuner:
  # 可选项: TPE, Random, Anneal, Evolution, BatchTuner
  #SMAC (SMAC 需要先通过 nnictl 来安装)
  builtinTunerName: TPE
  classArgs:
    # 可选项:: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
#local 模式下 machineList 可为空
machineList:
  - ip: 10.1.1.1
    username: bob
    passwd: bob123
    #使用默认端口 22 时，该配置可跳过
    #port: 22
  - ip: 10.1.1.2
    username: bob
    passwd: bob123
  - ip: 10.1.1.3
    username: bob
    passwd: bob123
```

`codeDir` 中的文件会自动上传到远程计算机中。 可在 Windows、Linux 或 macOS 上运行以下命令，在远程 Linux 计算机上启动 Trial：

```bash
nnictl create --config examples/trials/mnist-annotation/config_remote.yml
```

### Configure python environment

By default, commands and scripts will be executed in the default environment in remote machine. If there are multiple python virtual environments in your remote machine, and you want to run experiments in a specific environment, then use **preCommand** to specify a python environment on your remote machine.

Use `examples/trials/mnist-tfv2` as the example. Below is content of `examples/trials/mnist-tfv2/config_remote.yml`:

```yaml
authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: remote
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
#machineList can be empty if the platform is local
machineList:
  - ip: ${replace_to_your_remote_machine_ip}
    username: ${replace_to_your_remote_machine_username}
    sshKeyPath: ${replace_to_your_remote_machine_sshKeyPath}
    # Pre-command will be executed before the remote machine executes other commands.
    # Below is an example of specifying python environment.
    # If you want to execute multiple commands, please use "&&" to connect them.
    # preCommand: source ${replace_to_absolute_path_recommended_here}/bin/activate
    # preCommand: source ${replace_to_conda_path}/bin/activate ${replace_to_conda_env_name}
    preCommand: export PATH=${replace_to_python_environment_path_in_your_remote_machine}:$PATH
```

The **preCommand** will be executed before the remote machine executes other commands. So you can configure python environment path like this:

```yaml
# Linux remote machine
preCommand: export PATH=${replace_to_python_environment_path_in_your_remote_machine}:$PATH
# Windows remote machine
preCommand: set path=${replace_to_python_environment_path_in_your_remote_machine};%path%
```

Or if you want to activate the `virtualenv` environment:

```yaml
# Linux remote machine
preCommand: source ${replace_to_absolute_path_recommended_here}/bin/activate
# Windows remote machine
preCommand: ${replace_to_absolute_path_recommended_here}\\scripts\\activate
```

Or if you want to activate the `conda` environment:

```yaml
# Linux remote machine
preCommand: source ${replace_to_conda_path}/bin/activate ${replace_to_conda_env_name}
# Windows remote machine
preCommand: call activate ${replace_to_conda_env_name}
```

If you want multiple commands to be executed, you can use `&&` to connect these commands:

```yaml
preCommand: command1 && command2 && command3
```

**Note**: Because **preCommand** will execute before other commands each time, it is strongly not recommended to set **preCommand** that will make changes to system, i.e. `mkdir` or `touch`.