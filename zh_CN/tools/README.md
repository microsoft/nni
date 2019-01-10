## NNI CTL

NNI CTL 模块用来控制 Neural Network Intelligence，包括开始新实验，停止实验，更新实验等。

## 环境

    Ubuntu 16.04 或其它 Linux 操作系统。
    python >= 3.5
    

## 安装

1. 进入 tools 目录

2. 使用 pip 来安装包
    
    - 为当前用户安装：
        
        python3 -m pip install --user -e .
    
    - 为所有用户安装:
        
        python3 -m pip install -e .
    
    1. Change the mode of nnictl file 
    
    chmod +x ./nnictl
    
    2. Add nnictl to your PATH system environment variable. 
    - You could use `export` command to set PATH variable temporary.
        
        export PATH={your nnictl path}:$PATH
    
    - Or you could edit your `/etc/profile` file.
        
        1.sudo vim /etc/profile
        
        2.At the end of the file, add
        
              export PATH={your nnictl path}:$PATH
            
            save and exit.
            
        
        3.source /etc/profile

## 开始使用 NNI CTL

参考 [NNI CTL 文档](../docs/NNICTLDOC.md)。