# **实现 NNI TrainingService**

## 概述

TrainingService 是与平台管理、任务调度相关的模块。 TrainingService 在设计上为了便于实现，将平台相关的公共属性抽象成类。用户只需要继承这个抽象类，并根据平台特点实现子类，便能够实现 TrainingService。

## 系统架构

![](../img/NNIDesign.jpg)

NNI 的架构如图所示。 NNIManager 是系统的核心管理模块，负责调用 TrainingService 来管理 Trial，并负责不同模块之间的通信。 Dispatcher 是消息处理中心。 TrainingService 是管理任务的模块，它和 NNIManager 通信，并且根据平台的特点有不同的实现。 For the time being, NNI supports local platfrom, [remote platfrom](RemoteMachineMode.md), [PAI platfrom](PaiMode.md), [kubeflow platform](KubeflowMode.md) and [FrameworkController platfrom](FrameworkController.md). In this document, we introduce the brief design of TrainingService. If users want to add a new TrainingService instance, they just need to complete a child class to implement TrainingService, don't need to understand the code detail of NNIManager, Dispatcher or other modules.

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

**setClusterMetadata(key: string, value: string)** ClusterMetadata is the data related to platform details, for examples, the ClusterMetadata defined in remote machine server is:

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

**getClusterMetadata(key: string)** This function will return the metadata value according to the values, it could be left empty if users don't need to use it.

**submitTrialJob(form: JobApplicationForm)** SubmitTrialJob is a function to submit new trial jobs, users should generate a job instance in TrialJobDetail type. TrialJobDetail 定义如下：

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

**cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean)** If this function is called, the trial started by the platform should be canceled. 不同的平台有不同的取消作业的方式，这个方法应该根据不同平台的特点，实现相应的细节。

**updateTrialJob(trialJobId: string, form: JobApplicationForm)** This function is called to update the trial job's status, trial job's status should be detected according to different platform, and be updated to `RUNNING`, `SUCCEED`, `FAILED` etc.

**getTrialJob(trialJobId: string)** This function returns a trialJob detail instance according to trialJobId.

**listTrialJobs()** Users should put all of trial job detail information into a list, and return the list.

**addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)** NNI will hold an EventEmitter to get job metrics, if there is new job metrics detected, the EventEmitter will be triggered. 用户需要在这个方法中开始 EventEmitter。

**removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)** Close the EventEmitter.

**run()** The run() function is a main loop function in TrainingService, users could set a while loop to execute their logic code, and finish executing them when the experiment is stopped.

**cleanUp()** This function is called to clean up the environment when a experiment is stopped. 用户需要在这个方法中实现与平台相关的清除操作。

## TrialKeeper 工具

NNI 提供了 TrialKeeper 工具，用来帮助维护 Trial 任务。 可以在 `nni/tools/nni_trial_tool` 文件夹中找到 TrialKeeper 的源代码。 如果想要运行在云平台上，这是维护任务的好工具。 The running architecture of TrialKeeper is show as follow: ![](../img/trialkeeper.jpg) When users submit a trial job to cloud platform, they should wrap their trial command into TrialKeeper, and start a TrialKeeper process in cloud platform. 注意，TrialKeeper 在远程平台中使用 RESTful 服务来和 TrainingService 进行通信，用户需要在本地机器启动一个 RESTful 服务来接受 TrialKeeper 的请求。 关于 RESTful 服务的源代码可以在 `nni/src/nni_manager/training_service/common/clusterJobRestServer.ts` 文件夹中找到.

## 参考

For more information about how to debug, please [refer](HowToDebug.md). The guide line of how to contribute, please [refer](Contributing.md).