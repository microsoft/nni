# **实现 NNI TrainingService**

## 概述

TrainingService 是与平台管理、任务调度相关的模块。 TrainingService 在设计上为了便于实现，将平台相关的公共属性抽象成类。用户只需要继承这个抽象类，并根据平台特点实现子类，便能够实现 TrainingService。

## 系统架构

![](../img/NNIDesign.jpg)

NNI 的架构如图所示。 NNIManager 是系统的核心管理模块，负责调用 TrainingService 来管理 Trial，并负责不同模块之间的通信。 Dispatcher 是消息处理中心。 TrainingService 是管理任务的模块，它和 NNIManager 通信，并且根据平台的特点有不同的实现。 当前，NNI 支持本地平台、[远程平台](RemoteMachineMode.md)、[OpenPAI 平台](PaiMode.md)、[Kubeflow 平台](KubeflowMode.md)和[FrameworkController 平台](FrameworkController.md)。  
在这个文档中，会简要介绍 TrainingService 的设计。 如果要添加新的 TrainingService，只需要继承 TrainingServcie 类并实现相应的方法，不需要理解NNIManager、Dispatcher 等其它模块的细节。

## 代码文件夹结构

NNI 的文件夹结构如下：

    nni
      |- deployment
      |- docs
      |- examaples
      |- src
      | |- nni_manager
      | | |- common
      | | |- config
      | | |- core
      | | |- coverage
      | | |- dist
      | | |- rest_server
      | | |- training_service
      | | | |- common
      | | | |- kubernetes
      | | | |- local
      | | | |- pai
      | | | |- remote_machine
      | | | |- test
      | |- sdk
      | |- webui
      |- test
      |- tools
      | |-nni_annotation
      | |-nni_cmd
      | |-nni_gpu_tool
      | |-nni_trial_tool
    

`nni/src` 文件夹存储 NNI 的大部分源代码。 这个文件夹中的代码和 NNIManager、TrainingService、SDK、WebUI 等模块有关。 用户可以在 `nni/src/nni_manager/common/trainingService.ts` 文件中找到 TrainingService 抽象类的代码，并且把自己实现的子类放到 `nni/src/nni_manager/training_service` 文件夹下。 如果用户实现了自己的 TrainingService，还需要同时实现相应的单元测试代码，并把单元测试放到 `nni/src/nni_manager/training_service/test` 文件夹下。

## TrainingService 函数解释

    abstract class TrainingService {
        public abstract listTrialJobs(): Promise<TrialJobDetail[]>;
        public abstract getTrialJob(trialJobId: string): Promise<TrialJobDetail>;
        public abstract addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void;
        public abstract removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void): void;
        public abstract submitTrialJob(form: JobApplicationForm): Promise<TrialJobDetail>;
        public abstract updateTrialJob(trialJobId: string, form: JobApplicationForm): Promise<TrialJobDetail>;
        public abstract get isMultiPhaseJobSupported(): boolean;
        public abstract cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean): Promise<void>;
        public abstract setClusterMetadata(key: string, value: string): Promise<void>;
        public abstract getClusterMetadata(key: string): Promise<string>;
        public abstract cleanUp(): Promise<void>;
        public abstract run(): Promise<void>;
    }
    

TrainingService 父类有一些抽象方法，用户需要继承并实现这些抽象方法。

**setClusterMetadata(key: string, value: string)**  
ClusterMetadata 是与平台数据有关的方法，例如，在远程平台上的 ClusterMetadata 定义是：

    export class RemoteMachineMeta {
        public readonly ip : string;
        public readonly port : number;
        public readonly username : string;
        public readonly passwd?: string;
        public readonly sshKeyPath?: string;
        public readonly passphrase?: string;
        public gpuSummary : GPUSummary | undefined;
        /* GPU Reservation info, the key is GPU index, the value is the job id which reserves this GPU*/
        public gpuReservation : Map<number, string>;
    
        constructor(ip : string, port : number, username : string, passwd : string, 
            sshKeyPath : string, passphrase : string) {
            this.ip = ip;
            this.port = port;
            this.username = username;
            this.passwd = passwd;
            this.sshKeyPath = sshKeyPath;
            this.passphrase = passphrase;
            this.gpuReservation = new Map<number, string>();
        }
    }
    

Metadata 中包括了主机地址，用户名和其它平台相关配置。 用户需要定义自己的 Metadata 格式，并在这个方法中相应实现。 这个方法在 Experiment 启动之前调用。

**getClusterMetadata(key: string)**  
这个方法返回 metadata 的内容，如果不需要使用这个方法，可将方法内容留空。

**submitTrialJob(form: JobApplicationForm)**  
SubmitTrialJob 是用来提交 Trial 任务的方法，用户需要在这个方法中生成 TrialJobDetail 类型的实例。 TrialJobDetail 定义如下：

    interface TrialJobDetail {
        readonly id: string;
        readonly status: TrialJobStatus;
        readonly submitTime: number;
        readonly startTime?: number;
        readonly endTime?: number;
        readonly tags?: string[];
        readonly url?: string;
        readonly workingDirectory: string;
        readonly form: JobApplicationForm;
        readonly sequenceId: number;
        isEarlyStopped?: boolean;
    }
    

根据不同的实现，用户可能需要把 Trial 任务放入队列中，并不断地从队里中取出任务进行提交。 或者也可以直接在这个方法中完成作业提交过程。

**cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean)**  
如果这个方法被调用， Trial 应该被取消执行。 不同的平台有不同的取消作业的方式，这个方法应该根据不同平台的特点，实现相应的细节。

**updateTrialJob(trialJobId: string, form: JobApplicationForm)**  
这个方法用来更新 Trial 的状态，不同平台有不同的检测作业状态的方法，并把状态更新为`RUNNING`, `SUCCEED`, `FAILED` 等。

**getTrialJob(trialJobId: string)**  
这个方法用来根据 Trial Id 来返回相应的 Trial 实例。

**listTrialJobs()**  
用户需要在这个方法中把所有的 Trial 实例放入一个列表中，并返回。

**addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)**  
NNI 会启动一个 EventEmitter 来处理作业的指标数据，如果有检测到有新的数据，EventEmitter就会被触发，来执行相应的事件。 用户需要在这个方法中开始 EventEmitter。

**removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)**  
移除 EventEmitter。

**run()**  
Run() 函数是 TrainingService 的主循环，用户可以在这个函数中循环执行他们的代码逻辑，这个函数在实验结束前会一直循环执行。

**cleanUp()**  
当实验结束后，这个方法用来清除实验环境。 用户需要在这个方法中实现与平台相关的清除操作。

## TrialKeeper 工具

NNI 提供了 TrialKeeper 工具，用来帮助维护 Trial 任务。 可以在 `nni/tools/nni_trial_tool` 文件夹中找到 TrialKeeper 的源代码。 如果想要运行在云平台上，这是维护任务的好工具。 TrialKeeper 的架构如下：  
![](../img/trialkeeper.jpg)  
当用户需要在远程云平台上运行作业，要把作业启动的命令行传入 TrailKeeper 中，并在远程云平台上启动 TriakKeeper 进程。 注意，TrialKeeper 在远程平台中使用 RESTful 服务来和 TrainingService 进行通信，用户需要在本地机器启动一个 RESTful 服务来接受 TrialKeeper 的请求。 关于 RESTful 服务的源代码可以在 `nni/src/nni_manager/training_service/common/clusterJobRestServer.ts` 文件夹中找到.

## 参考

更多关于如何调试的信息，请[参考这里](HowToDebug.md)。  
关于如何贡献代码，请[参考这里](Contributing.md)。