# 训练平台

NNI 为 Trial 任务实现了训练平台。 NNI 支持[本机](./LocalMode.md), [远程](./RemoteMachineMode.md), [OpenPAI](./PaiMode.md), [Kubeflow](./KubeflowMode.md) 和 [FrameworkController](./FrameworkControllerMode.md) 这些内置的训练平台。  
NNI 不仅提供了这些内置的训练平台，还提供了轻松连接自己训练平台的方法。

## 内置训练平台

| 训练平台                                                    | 简介                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [**本机**](./LocalMode.md)                                | NNI 支持在本机运行实验，称为 local 模式。 local 模式表示 NNI 会在运行 NNI Manager 进程计算机上运行 Trial 任务，支持 GPU 调度功能。                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| [**远程计算机**](./RemoteMachineMode.md)                     | NNI 支持通过 SSH 通道在多台计算机上运行 Experiment，称为 remote 模式。 NNI 需要这些计算机的访问权限，并假定已配置好了深度学习训练环境。 NNI 将在远程计算机上中提交 Trial 任务，并根据 GPU 资源调度 Trial 任务。                                                                                                                                                                                                                                                                                                                                                                                                   |
| [**OpenPAI**](./PaiMode.md)                             | NNI 支持在 [OpenPAI](https://github.com/Microsoft/pai) （简称 pai）上运行 Experiment，即 pai 模式。 在使用 NNI 的 pai 模式前, 需要有 [OpenPAI](https://github.com/Microsoft/pai) 群集及其账户。 如果没有 OpenPAI，参考[这里](https://github.com/Microsoft/pai#how-to-deploy)来进行部署。 在 pai 模式中，会在 Docker 创建的容器中运行 Trial 程序。                                                                                                                                                                                                                                                       |
| [**Kubeflow**](./KubeflowMode.md)                       | NNI 支持在 [Kubeflow](https://github.com/kubeflow/kubeflow)上运行，称为 kubeflow 模式。 在开始使用 NNI 的 Kubeflow 模式前，需要有一个 Kubernetes 集群，可以是私有部署的，或者是 [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/zh-cn/services/kubernetes-service/)，并需要一台配置好 [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) 的 Ubuntu 计算机连接到此 Kubernetes 集群。 如果不熟悉 Kubernetes，可先浏览[这里](https://kubernetes.io/docs/tutorials/kubernetes-basics/)。 在 kubeflow 模式下，每个 Trial 程序会在 Kubernetes 集群中作为一个 Kubeflow 作业来运行。 |
| [**FrameworkController**](./FrameworkControllerMode.md) | NNI 支持使用 [FrameworkController](https://github.com/Microsoft/frameworkcontroller)，来运行 Experiment，称之为 frameworkcontroller 模式。 FrameworkController 构建于 Kubernetes 上，用于编排各种应用。这样，可以不用为某个深度学习框架安装 Kubeflow 的 tf-operator 或 pytorch-operator 等。 而直接用 FrameworkController 作为 NNI Experiment 的训练平台。                                                                                                                                                                                                                                            |


## 实现训练平台

TrainingService 在设计上为了便于实现，将平台相关的公共属性抽象成类。用户只需要继承这个抽象类，并根据平台特点实现子类，便能够实现 TrainingService。  
TrainingService 的声明如下：

```javascript
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
```

TrainingService 的父类有一些抽象函数，用户需要继承父类并实现所有这些抽象函数。  
有关如何实现 TrainingService 的更多信息，[参考这里](https://github.com/microsoft/nni/blob/master/docs/zh_CN/TrainingService/HowToImplementTrainingService.md)。