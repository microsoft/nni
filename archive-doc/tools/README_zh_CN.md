## NNI CTL

NNI CTL 模块用来控制 Neural Network Intelligence，包括开始新 Experiment，停止 Experiment，更新 Experiment等。

## 环境

    Ubuntu 16.04 或其它 Linux 操作系统。
    python >= 3.6
    

## 安装

1. 进入 tools 目录

2. 使用 pip 来安装包
    
    - 为当前用户安装：
        
        ```bash
        python3 -m pip install --user -e .
        ```
    
    - 为所有用户安装:
        
        ```bash
        python3 -m pip install -e .
        ```

3. 修改 nnictl 文件的权限
    
    ```bash
    chmod +x ./nnictl
    ```

4. 将 nnictl 添加到系统的 PATH 环境变量中。
    
    - 可以用 `export` 命令来临时设置 PATH 变量。
        
        export PATH={your nnictl path}:$PATH
    
    - 或者编辑 `/etc/profile` 文件。
        
        ```txt
        1.sudo vim /etc/profile
        
        2.在文件末尾加上
        
            export PATH={your nnictl path}:$PATH
        
        保存并退出。
        
        3.source /etc/profile
        ```

## 开始使用 NNI CTL

参考 [NNI CTL 文档](../docs/zh_CN/Nnictl.md)。