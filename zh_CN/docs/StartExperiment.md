# 如何启动 Experiment

## 1. 简介

NNI Experiment 的启动分为几步，流程如下。

<img src="./img/experiment_process.jpg" width="50%" height="50%" />

## 2. 详细说明

### 2.1 检查环境

1. 检查是否有旧的 Experiment 正在运行。 
2. 检查 RESTful 服务端口是否可用。 
3. 验证 yml 配置文件的内容。 
4. 准备配置文件来记录 Experiment 信息。 

### 2.2 启动 RESTful 服务

启动 RESTful 服务进程来管理 NNI Experiment，默认端口为 8080。

### 2.3 检查 RESTful 服务

检查是否 RESTful 服务进程成功启动，发送到 RESTful 服务的消息是否正常返回。

### 2.4 设置 Experiment 配置

在启动 Experiment 前调用 RESTful 服务来设置 Experiment 配置，包括 yml 配置文件中的值。

### 2.5 检查 Experiment 配置

检查 RESTful 服务的返回内容，如果状态为 200，则表示配置设置成功。

### 2.6 启动 Experiment

调用 RESTful 服务进程来设置 Experiment。

### 2.7 检查 Experiment

1. 检查 RESTful 服务的返回值。
2. 处理错误信息。
3. 输出成功或失败信息。
4. 保存配置信息到 nnictl 的配置文件。