**Run an Experiment on OpenPAI**
===
NNI supports running an experiment on [OpenPAI](https://github.com/Microsoft/pai) (aka pai), called pai mode. Before starting to use NNI pai mode, you should have an account to access an [OpenPAI](https://github.com/Microsoft/pai) cluster. See [here](https://github.com/Microsoft/pai#how-to-deploy) if you don't have any OpenPAI account and want to deploy an OpenPAI cluster. In pai mode, your trial program will run in pai's container created by Docker.

## Setup environment
Install NNI, follow the install guide [here](QuickStart.md).

## Run an experiment
Use `examples/trials/mnist-annotation` as an example. The NNI config YAML file's content is like:

```yaml
authorName: your_name
experimentName: auto_mnist
# how many trials could be concurrently running
trialConcurrency: 2
# maximum experiment running duration
maxExecDuration: 3h
# empty means never stop
maxTrialNum: 100
# choice: local, remote, pai
trainingServicePlatform: pai
# choice: true, false  
useAnnotation: true
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 mnist.py
  codeDir: ~/nni/examples/trials/mnist-annotation
  gpuNum: 0
  cpuNum: 1
  memoryMB: 8196
  image: msranni/nni:latest
  dataDir: hdfs://10.1.1.1:9000/nni
  outputDir: hdfs://10.1.1.1:9000/nni
# Configuration to access OpenPAI Cluster
paiConfig:
  userName: your_pai_nni_user
  passWord: your_pai_password
  host: 10.1.1.1
```

Note: You should set `trainingServicePlatform: pai` in NNI config YAML file if you want to start experiment in pai mode.

Compared with LocalMode and [RemoteMachineMode](RemoteMachineMode.md), trial configuration in pai mode have these additional keys:
* cpuNum
    * Required key. Should be positive number based on your trial program's CPU  requirement
* memoryMB
    * Required key. Should be positive number based on your trial program's memory requirement
* image
    * Required key. In pai mode, your trial program will be scheduled by OpenPAI to run in [Docker container](https://www.docker.com/). This key is used to specify the Docker image used to create the container in which your trial will run.
    * We already build a docker image [nnimsra/nni](https://hub.docker.com/r/msranni/nni/) on [Docker Hub](https://hub.docker.com/). It contains NNI python packages, Node modules and javascript artifact files required to start experiment, and all of NNI dependencies. The docker file used to build this image can be found at [here](https://github.com/Microsoft/nni/tree/master/deployment/docker/Dockerfile). You can either use this image directly in your config file, or build your own image based on it.
* dataDir
    * Optional key. It specifies the HDFS data direcotry for trial to download data. The format should be something like hdfs://{your HDFS host}:9000/{your data directory}
* outputDir 
    * Optional key. It specifies the HDFS output directory for trial. Once the trial is completed (either succeed or fail), trial's stdout, stderr will be copied to this directory by NNI sdk automatically. The format should be something like hdfs://{your HDFS host}:9000/{your output directory}
* virtualCluster
    * Optional key. Set the virtualCluster of OpenPAI. If omitted, the job will run on default virtual cluster.
* shmMB
    * Optional key. Set the shmMB configuration of OpenPAI, it set the shared memory for one task in the task role.

Once complete to fill NNI experiment config file and save (for example, save as exp_pai.yml), then run the following command
```
nnictl create --config exp_pai.yml
```
to start the experiment in pai mode. NNI will create OpenPAI job for each trial, and the job name format is something like `nni_exp_{experiment_id}_trial_{trial_id}`. 
You can see jobs created by NNI in the OpenPAI cluster's web portal, like:
![](../img/nni_pai_joblist.jpg)

Notice: In pai mode, NNIManager will start a rest server and listen on a port which is your NNI WebUI's port plus 1. For example, if your WebUI port is `8080`, the rest server will listen on `8081`, to receive metrics from trial job running in Kubernetes. So you should `enable 8081` TCP port in your firewall rule to allow incoming traffic. 

Once a trial job is completed, you can goto NNI WebUI's overview page (like http://localhost:8080/oview) to check trial's information. 

Expand a trial information in trial list view, click the logPath link like:
![](../img/nni_webui_joblist.jpg)

And you will be redirected to HDFS web portal to browse the output files of that trial in HDFS:
![](../img/nni_trial_hdfs_output.jpg)

You can see there're three fils in output folder: stderr, stdout, and trial.log

If you also want to save trial's other output into HDFS, like model files, you can use environment variable `NNI_OUTPUT_DIR` in your trial code to save your own output files, and NNI SDK will copy all the files in `NNI_OUTPUT_DIR` from trial's container to HDFS. 

Any problems when using NNI in pai mode, please create issues on [NNI github repo](https://github.com/Microsoft/nni).

## version check
NNI support version check feature in since version 0.6. It is a policy to insure the version of NNIManager is consistent with trialKeeper, and avoid errors caused by version incompatibility.  
Check policy:  
1. NNIManager before v0.6 could run any version of trialKeeper, trialKeeper support backward compatibility.
2. Since version 0.6, NNIManager version should keep same with triakKeeper version. For example, if NNIManager version is 0.6, trialKeeper version should be 0.6 too.  
3. Note that the version check feature only check first two digits of version.For example, NNIManager v0.6.1 could use trialKeeper v0.6 or trialKeeper v0.6.2, but could not use trialKeeper v0.5.1 or trialKeeper v0.7.  
  
If you could not run your experiment and want to know if it is caused by version check, you could check your webUI, and there will be an error message about version check.  
![](../img/version_check.png)