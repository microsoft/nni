# Experiment config reference
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
#choice: local, remote, pai
trainingServicePlatform: 
searchSpacePath: 
#choice: true, false
useAnnotation: 
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuNum: 
trial:
  command: 
  codeDir: 
  gpuNum: 
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
#choice: local, remote, pai
trainingServicePlatform: 
searchSpacePath: 
#choice: true, false
useAnnotation: 
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuNum: 
assessor:
  #choice: Medianstop
  builtinAssessorName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuNum: 
trial:
  command: 
  codeDir: 
  gpuNum: 
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
#choice: local, remote, pai
trainingServicePlatform: 
#choice: true, false
useAnnotation: 
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuNum: 
assessor:
  #choice: Medianstop
  builtinAssessorName:
  classArgs:
    #choice: maximize, minimize
    optimize_mode:
  gpuNum: 
trial:
  command: 
  codeDir: 
  gpuNum: 
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
   TBD: add default value
	 
* __experimentName__
  * Description
  
    __experimentName__ is the name of the experiment you created.  
    TBD: add default value
	
* __trialConcurrency__
  * Description
    
	 __trialConcurrency__ specifies the max num of trial jobs run simultaneously.  
	 
	    Note: if you set trialGpuNum bigger than the free gpu numbers in your machine, and the trial jobs running simultaneously can not reach trialConcurrency number, some trial jobs will be put into a queue to wait for gpu allocation.
	 
* __maxExecDuration__
  * Description
    
	__maxExecDuration__ specifies the max duration time of an experiment.The unit of the time is {__s__, __m__, __h__, __d__}, which means {_seconds_, _minutes_, _hours_, _days_}.  
	
* __maxTrialNum__
  *  Description
    
	 __maxTrialNum__ specifies the max number of trial jobs created by nni, including succeeded and failed jobs.  
	 
* __trainingServicePlatform__
  * Description
      
	  __trainingServicePlatform__ specifies the platform to run the experiment, including {__local__, __remote__}.  
	* __local__ mode means you run an experiment in your local linux machine.  
	
	* __remote__ mode means you submit trial jobs to remote linux machines. If you set platform as remote, you should complete __machineList__ field.  

	* __pai__ mode means you submit trial jobs to [OpenPai](https://github.com/Microsoft/pai) of Microsoft. For more details of pai configuration, please reference [PAIMOdeDoc](./PAIMode.md)
	
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
  
    __tuner__ specifies the tuner algorithm you use to run an experiment, there are two kinds of ways to set tuner. One way is to use tuner provided by nni sdk, you just need to set __builtinTunerName__ and __classArgs__. Another way is to use your own tuner file, and you need to set __codeDirectory__, __classFileName__, __className__ and __classArgs__.
  * __builtinTunerName__ and __classArgs__
    * __builtinTunerName__
    
	  __builtinTunerName__ specifies the name of system tuner you want to use, nni sdk provides four kinds of tuner, including {__TPE__, __Random__, __Anneal__, __Evolution__}
	 * __classArgs__
	
	   __classArgs__ specifies the arguments of tuner algorithm
  * __codeDir__, __classFileName__, __className__ and __classArgs__
      * __codeDir__
        
		__codeDir__ specifies the directory of tuner code.
	    * __classFileName__
	   
	  __classFileName__ specifies the name of tuner file.
     * __className__
	   
	  __className__ specifies the name of tuner class.
     * __classArgs__
	   
	  __classArgs__ specifies the arguments of tuner algorithm.
  * __gpuNum__
    
	  __gpuNum__ specifies the gpu number you want to use to run the tuner process. The value of this field should be a positive number.
	  
	    Note: you could only specify one way to set tuner, for example, you could set {tunerName, optimizationMode} or {tunerCommand, tunerCwd}, and you could not set them both. 

* __assessor__
 
  * Description
  
    __assessor__ specifies the assessor algorithm you use to run an experiment, there are two kinds of ways to set assessor. One way is to use assessor provided by nni sdk, you just need to set __builtinAssessorName__ and __classArgs__. Another way is to use your own tuner file, and you need to set __codeDirectory__, __classFileName__, __className__ and __classArgs__.
  * __builtinAssessorName__ and __classArgs__
    * __builtinAssessorName__
    
	  __builtinAssessorName__ specifies the name of system assessor you want to use, nni sdk provides four kinds of tuner, including {__TPE__, __Random__, __Anneal__, __Evolution__}
	 * __classArgs__
	
	   __classArgs__ specifies the arguments of tuner algorithm
  * __codeDir__, __classFileName__, __className__ and __classArgs__
      * __codeDir__
        
		__codeDir__ specifies the directory of tuner code.
	    * __classFileName__
	   
	  __classFileName__ specifies the name of tuner file.
     * __className__
	   
	  __className__ specifies the name of tuner class.
     * __classArgs__
	   
	  __classArgs__ specifies the arguments of tuner algorithm.
  * __gpuNum__
    
	__gpuNum__ specifies the gpu number you want to use to run the assessor process. The value of this field should be a positive number.

        Note: you could only specify one way to set assessor, for example, you could set {assessorName, optimizationMode} or {assessorCommand, assessorCwd}, and you could not set them both.If you do not want to use assessor, you just need to leave assessor empty or remove assessor in your config file. Default value is 0. 
* __trial__
  * __command__

      __command__  specifies the command to run trial process.
  * __codeDir__
    
	  __codeDir__ specifies the directory of your own trial file.
  * __gpuNum__
    
	  __gpuNum__ specifies the num of gpu you want to use to run your trial process. Default value is 0. 
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

  * __sshKeyPath__

    If you want to use ssh key to login remote machine, you could set __sshKeyPath__ in config file. __sshKeyPath__ is the path of ssh key file, which should be valid.
	
	    Note: if you set passwd and sshKeyPath simultaneously, nni will try passwd.
		
  * __passphrase__

    __passphrase__ is used to protect ssh key, which could be empty if you don't have passphrase.


## Examples
* __local mode__

  If you want to run your trial jobs in your local machine, and use annotation to generate search space, you could use the following config:
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
#choice: true, false
useAnnotation: true
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
  gpuNum: 0
trial:
  command: python3 mnist.py
  codeDir: /nni/mnist
  gpuNum: 0
```

  If you want to use assessor, you could add assessor configuration in your file.
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: /nni/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
  gpuNum: 0
assessor:
  #choice: Medianstop
  builtinAssessorName: Medianstop
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
  gpuNum: 0
trial:
  command: python3 mnist.py
  codeDir: /nni/mnist
  gpuNum: 0
```

  Or you could specify your own tuner and assessor file as following:
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
searchSpacePath: /nni/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  codeDir: /nni/tuner
  classFileName: mytuner.py
  className: MyTuner
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
  gpuNum: 0
assessor:
  codeDir: /nni/assessor
  classFileName: myassessor.py
  className: MyAssessor
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
  gpuNum: 0
trial:
  command: python3 mnist.py
  codeDir: /nni/mnist
  gpuNum: 0
```

* __remote mode__

If you want run trial jobs in your remote machine, you could specify the remote mahcine information as fllowing format:
```
authorName: test
experimentName: test_experiment
trialConcurrency: 3
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: remote
searchSpacePath: /nni/search_space.json
#choice: true, false
useAnnotation: false
tuner:
  #choice: TPE, Random, Anneal, Evolution
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
  gpuNum: 0
trial:
  command: python3 mnist.py
  codeDir: /nni/mnist
  gpuNum: 0
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
    sshKeyPath: /nni/sshkey
    passphrase: qwert
```
