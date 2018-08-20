**Enable Assessor in your expeirment**
===
Assessor module is for assessing running trials. One common use case is early stopping, which terminates unpromising trial jobs based on their intermediate results.

## Using NNI built-in Assessor
Here we use the same example `examples/trials/mnist-annotation`. We use `Medianstop` assessor for this experiment. The yaml configure file is shown below:
```
authorName: your_name
experimentName: auto_mnist
# how many trials could be concurrently running
trialConcurrency: 2
# maximum experiment running duration
maxExecDuration: 3h
# empty means never stop
maxTrialNum: 100
# choice: local, remote  
trainingServicePlatform: local
# choice: true, false  
useAnnotation: true
tuner:
  tunerName: TPE
  optimizationMode: Maximize
assessor:
  assessorName: Medianstop
  optimizationMode: Maximize
trial:
  trialCommand: python mnist.py
  trialCodeDir: /usr/share/nni/examples/trials/mnist-annotation
  trialGpuNum: 0
```
For our built-in assessors, you need to fill two fields: `assessorName` which chooses NNI provided assessors (refer to [here]() for built-in assessors), `optimizationMode` which includes Maximize and Minimize (you want to maximize or minimize your trial result).

## Using user customized Assessor
You can also write your own assessor following the guidance [here](). For example, you wrote an assessor for `examples/trials/mnist-annotation`. You should prepare the yaml configure below:
```
authorName: your_name
experimentName: auto_mnist
# how many trials could be concurrently running
trialConcurrency: 2
# maximum experiment running duration
maxExecDuration: 3h
# empty means never stop
maxTrialNum: 100
# choice: local, remote  
trainingServicePlatform: local
# choice: true, false  
useAnnotation: true
tuner:
  tunerName: TPE
  optimizationMode: Maximize
assessor:
  assessorCommand: your_command
  assessorCodeDir: /path/of/your/asessor
  assessorGpuNum: 0
trial:
  trialCommand: python mnist.py
  trialCodeDir: /usr/share/nni/examples/trials/mnist-annotation
  trialGpuNum: 0
```
You only need to fill three field: `assessorCommand`, `assessorCodeDir` and `assessorGpuNum`.