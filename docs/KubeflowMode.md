**Run an Experiment on Kubeflow**
===
Now NNI supports running experiment on [Kubeflow](https://github.com/kubeflow/kubeflow), called kubeflow mode. Before starting to use NNI kubeflow mode, you should have a kubernetes cluster, either on-prem or [Azure Kubernetes Service(AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/), a Ubuntu machine on which [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl/) is installed and configured to connect to your kubernetes cluster. If you are not familiar with kubernetes, [here](https://kubernetes.io/docs/tutorials/kubernetes-basics/) is a goot start. In kubeflow mode, your trial program will run as kubeflow job in kubernetes cluster. 

## Prerequisite
1. A **Kubernetes** cluster using Kubernetes 1.8 or later. Follow this [guideline](https://kubernetes.io/docs/setup/) to set up Kubernetes
2. Download, set up, and deploy **Kubelow** to your Kubernetes cluster. Follow this [guideline](https://www.kubeflow.org/docs/started/getting-started/) to set up Kubeflow
3. Install **kubectl**, and configure to connect to your Kubernetes API server.
4. If your NNI trial job needs GPU resource, you should follow this [guideline](https://github.com/NVIDIA/k8s-device-plugin) to configure **Nvidia device plugin for Kubernetes**.
5. An **NFS** server and export a general purpose mount (we recommend to map your NFS server path in root_squash option. Refer this [page](https://linux.die.net/man/5/exports) to learn what root_squash option is), or **Azure File Storage**. 
6. Install **NNI**, follow the install guide [here](GetStarted.md).

## Design 
TODO

## Run an experiment
Use `examples/trials/mnist` as an example. The nni config yaml file's content is like: 
```
authorName: your_name
experimentName: example_mnist
# how many trials could be concurrently running
trialConcurrency: 4
# maximum experiment running duration
maxExecDuration: 3h
# empty means never stop
maxTrialNum: 100
# choice: local, remote, pai, kubeflow
trainingServicePlatform: kubeflow
# choice: true, false  
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
trial:
  codeDir: ~/nni/examples/trials/mnist
  ps:
    replicas: 1 
    command: python mnist-keras.py    
    gpuNum: 0
    cpuNum: 1
    memoryMB: 8196
    image: {your_docker_image_for_tensorflow_ps}
  worker:
    replicas: 1 
    command: python mnist-keras.py    
    gpuNum: 2
    cpuNum: 1
    memoryMB: 8196
    image: {your_docker_image_for_tensorflow_worker}
kubeflowConfig:
  operator: tf-operator
  nfs:
    server: {your_nfs_server}
    path: {your_nfs_server_exported_path}
  kubernetesServer: {your_kubernetes_api_server_ip}
```
Note: You should explicitly set `trainingServicePlatform: kubeflow` in nni config yaml file if you want to start experiment in kubeflow mode. 

Trial configuration in kubeflow mode have the following configuration keys:
* codeDir
    * code directory, where you put training code and config files
* worker (required). This config section is used to configure tensorflow worker role
    * replicas
        * Required key. Should be positive number depends on how many replication your want to run for tensorflow worker role.
    * command
        * Required key. Command to launch your trial job, like ```python mnist.py```
    * memoryMB
        * Required key. Should be positive number based on your trial program's memory requirement
    * cpuNum
    * gpuNum
    * image
        * Required key. In kubeflow mode, your trial program will be scheduled by Kubernetes to run in [Pod](https://kubernetes.io/docs/concepts/workloads/pods/pod/). This key is used to specify the Docker image used to create the pod where your trail program will run. 
        * We already build a docker image [nnimsra/nni](https://hub.docker.com/r/msranni/nni/) on [Docker Hub](https://hub.docker.com/). It contains NNI python packages, Node modules and javascript artifact files required to start experiment, and all of NNI dependencies. The docker file used to build this image can be found at [here](../deployment/Dockerfile.build.base). You can either use this image directly in your config file, or build your own image based on it.
* ps (optional). This config section is used to configure tensorflow parameter server role.

Once complete to fill nni experiment config file and save (for example, save as exp_kubeflow.yaml), then run the following command
```
nnictl create --config exp_kubeflow.yaml
```
to start the experiment in kubeflow mode. NNI will create Kubeflow tfjob for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. 
You can see the kubeflow tfjob created by NNI in your Kubernetes dashboard. 

Notice: In kubeflow mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic. 

Once a trial job is completed, you can goto NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information. 

Any problems when using NNI in kubeflow mode, plesae create issues on [NNI github repo](https://github.com/Microsoft/nni), or send mail to nni@microsoft.com

