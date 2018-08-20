Experiment config reference
===

If you want to create a new nni experiment, you need to prepare a config file in your local machine, and provide the path of this file to nnictl.
The config file is written in yaml format, and need to be written correctly.
This document describes the rule to write config file, and will provide some examples and templates for you. 
## Template
* __light weight(without Annotation and Assessor)__ 
```
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#choice: local, remote
trainingServicePlatform: 
searchSpacePath: 
#choice: true, false
useAnnotation: 
tuner:
  #choice: TPE, Random, Anneal, Evolution
  tunerName: 
  #choice: Maximize, Minimize
  optimizationMode: 
  tunerGpuNum: 
trial:
  trialCommand: 
  trialCodeDir: 
  trialGpuNum: 
#machineList can be empty if the platform is local
machineList:
  - ip: 
    port: 
    username: 
    passwd: 
```
* __Use Assessor__
```
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#choice: local, remote
trainingServicePlatform: 
searchSpacePath: 
#choice: true, false
useAnnotation: 
tuner:
  #choice: TPE, Random, Anneal, Evolution
  tunerName: 
  #choice: Maximize, Minimize
  optimizationMode: 
  tunerGpuNum: 
assessor:
  #choice: Medianstop
  assessorName: 
  #choice: Maximize, Minimize
  optimizationMode: 
  assessorGpuNum: 
trial:
  trialCommand: 
  trialCodeDir: 
  trialGpuNum: 
#machineList can be empty if the platform is local
machineList:
  - ip: 
    port: 
    username: 
    passwd: 
```
* __Use Annotation__
```
authorName: 
experimentName: 
trialConcurrency: 
maxExecDuration: 
maxTrialNum: 
#choice: local, remote
trainingServicePlatform: 
#choice: true, false
useAnnotation: 
tuner:
  #choice: TPE, Random, Anneal, Evolution
  tunerName: 
  #choice: Maximize, Minimize
  optimizationMode: 
  tunerGpuNum: 
assessor:
  #choice: Medianstop
  assessorName: 
  #choice: Maximize, Minimize
  optimizationMode: 
  assessorGpuNum: 
trial:
  trialCommand: 
  trialCodeDir: 
  trialGpuNum: 
#machineList can be empty if the platform is local
machineList:
  - ip: 
    port: 
    username: 
    passwd: 
```
## Configuration
* __authorName__
  * Description  
            
	 __authorName__ is the name of the author who create the experiment.
	 
* __experimentName__
  * Description
  
    __experimentName__ is the name of the experiment you created.  
	
* __trialConcurrency__
  * Description
    
	 __trialConcurrency__ specifies the max num of trial jobs run simultaneously.  
	 
	    Note: if you set trialGpuNum bigger than the free gpu numbers in your machine, and the trial jobs running simultaneously can not reach trialConcurrency number, some trial jobs will be put into a queue to wait for gpu allocation.
	 
* __maxExecDuration__
  * Description
    
	__maxExecDuration__ specifies the max duration time of an experiment.The unit of the time is {__s__, __m__, __h__, __d__}, which means {_seconds_, _minutes_, _hours_, _days_}.  
	
* __maxTrialNum__
  *  Description
    
	 __maxTrialNum__ specifies the max number of trial jobs created by nni, including successed and failed jobs.  
	 
* __trainingServicePlatform__
  * Description
      
	  __trainingServicePlatform__ specifies the platform to run the experiment, including {__local__, __remote__}.  
	* __local__ mode means you run an experiment in your local linux machine.  
	
	* __remote__ mode means you submit trial jobs to remote linux machines. If you set platform as remote, you should complete __machineList__ field.  
	
* __searchSpacePath__
  * Description
    
	 __searchSpacePath__ specifies the path of search space file you want to use, which should be a valid path in your local linux machine.
	        
	    Note: if you set useAnnotation=True, you should remove searchSpacePath field or just let it be empty.
* __useAnnotation__
  * Description
   
    __useAnnotation__ means whether you use annotation to analysis your code and generate search space. 
	   
	    Note: if you set useAnnotation=True, you should not set searchSpacePath.
		
* __tuner__
  * Description
  
    __tuner__ specifies the tuner algorithm you use to run an experiment, there are two kinds of ways to set tuner. One way is to use tuner provided by nni sdk, you just need to set __tunerName__ and __optimizationMode__. Another way is to use your own tuner file, and you need to set __tunerCommand__, __tunerCwd__.
  * __tunerName__ and __optimizationMode__
    * __tunerName__
    
	  __tunerName__ specifies the name of system tuner you want to use, nni sdk provides four kinds of tuner, including {__TPE__, __Random__, __Anneal__, __Evolution__}
	 * __optimizationMode__
	
	   __optimizationMode__ specifies the optimization mode of tuner algorithm, including {__Maximize__, __Minimize__}
  * __tunerCommand__ and __tunerCwd__
      * __tunerCommand__
        
		__tunerCommand__ specifies the command you want to use to run your own tuner file, for example {__python3 mytuner.py__}
	   * __tunerCwd__
	   
	     __tunerCwd__ specifies the working directory of your own tuner file, which is the path of your own tuner file.
  * __tunerGpuNum__
    
	  __tunerGPUNum__ specifies the gpu number you want to use to run the tuner process. The value of this field should be a positive number.
	  
	    Note: you could only specify one way to set tuner, for example, you could set {tunerName, optimizationMode} or {tunerCommand, tunerCwd}, and you could not set them both. 

* __assessor__
 
  * Description
 
    __assessor__ specifies the assessor algorithm you use to run experiment, there are two kinds of ways to set assessor. One way is to use assessor provided by nni sdk, you just need to set __assessorName__ and __optimizationMode__. Another way is to use your own assessor file, and you need to set __assessorCommand__, __assessorCwd__.
  * __assessorName__ and __optimizationMode__
    * __assessorName__
    
	  __assessorName__ specifies the name of system assessor you want to use, nni sdk provides one kind of assessor, which is {__Medianstop__}.
	 * __optimizationMode__
	 
	   __optimizationMode__ specifies the optimization mode of tuner algorithm, including {__Maximize__, __Minimize__}
  * __assessorCommand__ and __assessorCwd__
      * __assessorCommand__
        
		__assessorCommand__ specifies the command you want to use to run your own assessor file, for example {__python3 myassessor.py__}
	 * __assessorCwd__
	  
	   __assessorCwd__ specifies the working directory of your own assessor file, which is the path of your own assessor file.
  * __assessorGpuNum__
    
	__assessorGPUNum__ specifies the gpu number you want to use to run the assessor process. The value of this field should be a positive number.

        Note: you could only specify one way to set assessor, for example, you could set {assessorName, optimizationMode} or {assessorCommand, assessorCwd}, and you could not set them both.If you do not want to use assessor, you just need to leave assessor empty or remove assessor in your config file. 
* __trial__
  * __trialCommand__

      __trialCommand__  specifies the command to run trial process.
  * __trialCodeDir__
    
	  __trialCodeDir__ specifies the directory of your own trial file.
  * __trialGpuNum__
    
	  __trialGpuNum__ specifies the num of gpu you want to use to run your trial process.
* __machineList__
 
     __machineList__ should be set if you set __trainingServicePlatform__=remote, or it could be empty.
  * __ip__
    
	__ip__ is the ip address of your remote machine.
  * __port__
    
	__port__ is the ssh port you want to use to connect machine.
	
	    Note: if you set port empty, the default value will be 22.
  * __username__
    
	__username__ is the account you use.
  * __passwd__
    
	__passwd__ specifies the password of your account.

## Examples
* __local mode__

  If you want to run your trial jobs in your local machine, and use annotation to generate search space, you could use the following config:
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote
trainingServicePlatform: local
#choice: true, false
useAnnotation: true
tuner:
  #choice: TPE, Random, Anneal, Evolution
  tunerName: TPE
  #choice: Maximize, Minimize
  optimizationMode: Maximize
  tunerGpuNum: 0
trial:
  trialCommand: python3 mnist.py
  trialCodeDir: /nni/mnist
  trialGpuNum: 0
```

  If you want to use assessor, you could add assessor configuration in your file.
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote
trainingServicePlatform: local
searchSpacePath: /nni/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution
  tunerName: TPE
  #choice: Maximize, Minimize
  optimizationMode: Maximize
  tunerGpuNum: 0
assessor:
  #choice: Medianstop
  assessorName: Medianstop
  #choice: Maximize, Minimize
  optimizationMode: Maximize
  assessorGpuNum: 0
trial:
  trialCommand: python3 mnist.py
  trialCodeDir: /nni/mnist
  trialGpuNum: 0
```

  Or you could specify your own tuner and assessor file as following:
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote
trainingServicePlatform: local
searchSpacePath: /nni/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  tunerCommand: python3 mytuner.py
  tunerCwd: /nni/tuner
  tunerGpuNum: 0
assessor:
  assessorCommand: python3 myassessor.py
  assessorCwd: /nni/assessor
  assessorGpuNum: 0
trial:
  trialCommand: python3 mnist.py
  trialCodeDir: /nni/mnist
  trialGpuNum: 0
```

* __remote mode__

If you want run trial jobs in your remote machine, you could specify the remote mahcine information as fllowing format:
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote
trainingServicePlatform: remote
searchSpacePath: /nni/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution
  tunerName: TPE
  #choice: Maximize, Minimize
  optimizationMode: Maximize
  tunerGpuNum: 0
trial:
  trialCommand: python3 mnist.py
  trialCodeDir: /nni/mnist
  trialGpuNum: 0
#machineList can be empty if the platform is local
machineList:
  - ip: 10.10.10.10
    port: 22
    username: test
    passwd: test
  - ip: 10.10.10.11
    port: 22
    username: test
    passwd: test
  - ip: 10.10.10.12
    port: 22
    username: test
    passwd: test
```