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
    

`nni/src`文件夹存储NNI相关的大部分源代码。 The code in this folder is related to NNIManager, TrainingService, SDK, WebUI and other modules. Users could find the abstract class of TrainingService in `nni/src/nni_manager/common/trainingService.ts` file, and they should put their own implemented TrainingService in `nni/src/nni_manager/training_service` folder. If users have implemented their own TrainingService code, they should also supplement the unit test of the code, and place them in `nni/src/nni_manager/training_service/test` folder.

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
    

The parent class of TrainingService has a few abstract functions, users need to inherit the parent class and implement all of these abstract functions.

**setClusterMetadata(key: string, value: string)**  
ClusterMetadata is the data related to platform details, for examples, the ClusterMetadata defined in remote machine server is:

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
    

The metadata includes the host address, the username or other configuration related to the platform. Users need to define their own metadata format, and set the metadata instance in this function. This function is called before the experiment is started to set the configuration of remote machines.

**getClusterMetadata(key: string)**  
This function will return the metadata value according to the values, it could be left empty if users don't need to use it.

**submitTrialJob(form: JobApplicationForm)**  
SubmitTrialJob is a function to submit new trial jobs, users should generate a job instance in TrialJobDetail type. TrialJobDetail is defined as follow:

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
    

According to different kinds of implementation, users could put the job detail into a job queue, and keep fetching the job from the queue and start preparing and running them. Or they could finish preparing and running process in this function, and return job detail after the submit work.

**cancelTrialJob(trialJobId: string, isEarlyStopped?: boolean)**  
If this function is called, the trial started by the platform should be canceled. Different kind of platform has diffenent methods to calcel a running job, this function should be implemented according to specific platform.

**updateTrialJob(trialJobId: string, form: JobApplicationForm)**  
This function is called to update the trial job's status, trial job's status should be detected according to different platform, and be updated to `RUNNING`, `SUCCEED`, `FAILED` etc.

**getTrialJob(trialJobId: string)**  
This function returns a trialJob detail instance according to trialJobId.

**listTrialJobs()**  
Users should put all of trial job detail information into a list, and return the list.

**addTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)**  
NNI will hold an EventEmitter to get job metrics, if there is new job metrics detected, the EventEmitter will be triggered. Users should start the EventEmitter in this function.

**removeTrialJobMetricListener(listener: (metric: TrialJobMetric) => void)**  
Close the EventEmitter.

**run()**  
The run() function is a main loop function in TrainingService, users could set a while loop to execute their logic code, and finish executing them when the experiment is stopped.

**cleanUp()**  
This function is called to clean up the environment when a experiment is stopped. Users should do the platform-related cleaning operation in this function.

## TrialKeeper 工具

NNI offers a TrialKeeper tool to help maintaining trial jobs. Users can find the source code in `nni/tools/nni_trial_tool`. If users want to run trial jobs in cloud platform, this tool will be a fine choice to help keeping trial running in the platform. The running architecture of TrialKeeper is show as follow:  
![](../img/trialkeeper.jpg)  
When users submit a trial job to cloud platform, they should wrap their trial command into TrialKeeper, and start a TrialKeeper process in cloud platform. Notice that TrialKeeper use restful server to communicate with TrainingService, users should start a restful server in local machine to receive metrics sent from TrialKeeper. The source code about restful server could be found in `nni/src/nni_manager/training_service/common/clusterJobRestServer.ts`.

## 参考

For more information about how to debug, please [refer](HowToDebug.md).  
The guide line of how to contribute, please [refer](CONTRIBUTING).