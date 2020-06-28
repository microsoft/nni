# Training Service

NNI Training service provides the training platform for running NNI trial jobs. NNI supports [local](./LocalMode.md), [remote](./RemoteMachineMode.md), [PAI](./PaiMode.md), [Kubeflow](./KubeflowMode.md) and [FrameworkController](./FrameworkControllerMode.md) built-in training services.
NNI not only provides few built-in training service options, but also provides a method for customers to build their own training service easily.

## Built-in TrainingService

|TrainingService|Brief Introduction|
|---|---|
|[__Local__](./LocalMode.md)|NNI supports running an experiment on local machine, called local mode. Local mode means that NNI will run the trial jobs and nniManager process in same machine, and support gpu schedule function for trial jobs.|
|[__Remote__](./RemoteMachineMode.md)|NNI supports running an experiment on multiple machines through SSH channel, called remote mode. NNI assumes that you have access to those machines, and already setup the environment for running deep learning training code. NNI will submit the trial jobs in remote machine, and schedule suitable machine with enough gpu resource if specified.|
|[__Pai__](./PaiMode.md)|NNI supports running an experiment on [OpenPAI](https://github.com/Microsoft/pai) (aka pai), called pai mode. Before starting to use NNI pai mode, you should have an account to access an [OpenPAI](https://github.com/Microsoft/pai) cluster. See [here](https://github.com/Microsoft/pai#how-to-deploy) if you don't have any OpenPAI account and want to deploy an OpenPAI cluster. In pai mode, your trial program will run in pai's container created by Docker.|
|[__Kubeflow__](./KubeflowMode.md)|NNI supports running experiment on [Kubeflow](https://github.com/kubeflow/kubeflow), called kubeflow mode. Before starting to use NNI kubeflow mode, you should have a Kubernetes cluster, either on-premises or [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/), a Ubuntu machine on which [kubeconfig](https://kubernetes.io/docs/concepts/configuration/organize-cluster-access-kubeconfig/) is setup to connect to your Kubernetes cluster. If you are not familiar with Kubernetes, [here](https://kubernetes.io/docs/tutorials/kubernetes-basics/) is a good start. In kubeflow mode, your trial program will run as Kubeflow job in Kubernetes cluster.|
|[__FrameworkController__](./FrameworkControllerMode.md)|NNI supports running experiment using [FrameworkController](https://github.com/Microsoft/frameworkcontroller), called frameworkcontroller mode. FrameworkController is built to orchestrate all kinds of applications on Kubernetes, you don't need to install Kubeflow for specific deep learning framework like tf-operator or pytorch-operator. Now you can use FrameworkController as the training service to run NNI experiment.|

## TrainingService Implementation

TrainingService is designed to be easily implemented, we define an abstract class TrainingService as the parent class of all kinds of TrainingService, users just need to inherit the parent class and complete their own child class if they want to implement customized TrainingService.  
The abstract function in TrainingService is shown below:

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

The parent class of TrainingService has a few abstract functions, users need to inherit the parent class and implement all of these abstract functions.  
For more information about how to write your own TrainingService, please [refer](https://github.com/microsoft/nni/blob/master/docs/en_US/TrainingService/HowToImplementTrainingService.md).
