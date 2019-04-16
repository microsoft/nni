# **实现 NNI TrainingService**

## 概述

TrainingService是一个与平台和作业管理相关的模块。 TrainingService在设计上为了易于实现，把各种平台相关的公共属性抽象出来，构成了一个抽象类，用户只需要继承这个抽象的父类，并根据平台特点实现子类，便能够实现TrainingService。

## 系统架构

![](../img/NNIDesign.jpg)

NNI的架构如图所示。 NNIManager是系统的核心管理模块，负责调用TrainingService来管理Trial作业，并负责不同模块之间的通信。 Dispatcher是消息处理中心。 TrainingService是一个管理作业的模块，它和NNIManager进行通信，并且根据平台的特点有不同的实例。 现在，NNI支持本地平台、[远程平台](RemoteMachineMode.md)、[PAI平台](PAIMode.md)、[Kubeflow平台](KubeflowMode.md)和[FrameworkController平台](FrameworkController.md)。  
在这个文档中，简要介绍TrainingService的实现。 如果用户想要添加一个新的TrainingService，他们只需要继承TrainingServcie父类并实现相应的方法，不需要理解NNIManager、Dispatcher等其他模块的内容。

## 代码文件夹结构

NNI的文件夹结构如下：

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
    

`nni/src`文件夹存储NNI相关的大部分源代码。 在这个文件夹中的代码和NNIManager、TrainingService、SDK、WebUI等模块有关。 用户可以在`nni/src/nni_manager/common/trainingService.ts`文件中找到TrainingService抽象类的代码，并且把自己实现的子类放到 `nni/src/nni_manager/training_service`文件夹下。 如果用户实现了自己的TrainingService代码，也需要同时实现相应的单元测试代码，并把单元测试放到`nni/src/nni_manager/training_service/test` 文件夹下。

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
    

TrainingService父类有一些抽象方法，用户需要继承父类并实现这些抽象方法。

**setClusterMetadata(key: string, value: string)**  
CllusterMetadata是与平台数据有关的方法，例如，在远程平台上的ClusterMetadata定义是：

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
    

Metadata中包括了主机地址，用户名和其他平台相关配置。 用户需要定义他们自己的metadata格式，并在这个方法中设置相应的内容。 这个方法在实验启动之前调用。

**getClusterMetadata(key: string)**  
这个方法返回metadata的内容，如果用户不需要使用这个方法的话，可以把方法内容设置为空。

**submitTrialJob(form: JobApplicationForm)**  
SubmitTrialJob是用来提交Trial任务的方法，用户需要在这个方法中生成TrialJobDetail类型的Trial实例。 TrialJobDetail定义如下：

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
    

根据不同的实现，用户可能需要把Trial作业放入队列中，并不断地从队里中取出任务进行提交。 或者也可以直接在这个方法中完成作业提交过程。

**cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean)**  
如果这个方法被调用，trial应该被取消执行。 不同的平台有不同的取消作业的方式，这个方法应该根据不同平台的特点，实现相应的细节。

**updateTrialJob(trialJobId: string, form: JobApplicationForm)**  
这个方法用来更新Trial的作业状态，不同平台有不同的检测作业状态的方法，并把状态更新为`RUNNING`, `SUCCEED`, `FAILED` 等。

**getTrialJob(trialJobId: string)**  
这个方法用来根据TrialJobId来返回相应的Trial实例。

**listTrialJobs()**  
用户需要在这个方法中把所有的Trial作业实例放入一个list中，并返回这个list。

**addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)**  
NNI会启动一个EventEmitter来处理作业的metrics数据，如果有检测到有新的数据，EventEmitter就会被触发，来执行相应的事件。 用户需要在这个方法中设置EventEmitter。

**removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)**  
关闭 EventEmitter.

**run()**  
Run() 函数是TrainingService的主循环，用户可以在这个函数中循环执行他们的代码逻辑，这个函数在实验结束前会一直循环执行。

**cleanUp()**  
当实验结束后，这个方法用来清除实验环境。 用户需要在这个方法中实现与平台相关的清除操作。

## TrialKeeper 工具

NNI提供了一个TrialKeeper工具，用来帮助维护Trial作业。 用户可以在`nni/tools/nni_trial_tool`文件夹中找到TrialKeeper的源代码。 如果用户想要把作业运行在云平台上，这个工具对于维护作业是一个好的选择。 The running architecture of TrialKeeper is show as follow:  
![](../img/trialkeeper.jpg)  
当用户需要在远程云平台上运行作业，他们需要把作业启动的命令行传入TrailKeeper中，并在远程云平台上启动TriakKeeper进程。 注意，TrialKeeper在远程平台中使用restful服务来和TrainingService进行通信，用户需要在本地机器启动一个Restful服务来接受TrialKeeper的请求。 关于Restful服务的源代码可以在`nni/src/nni_manager/training_service/common/clusterJobRestServer.ts`文件夹中找到.

## 参考

更多关于如何debug的信息，请[参考](HowToDebug.md).  
。 关于如何贡献代码，请 [参考](CONTRIBUTING).