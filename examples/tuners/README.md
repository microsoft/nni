# Customized Tuner for Experts

*Tuner receive result from Trial as a matric to evaluate the performance of a specific parameters/architecture configure. And tuner send next hyper-parameter or architecture configure to Trial.*

So, if user want to implement a customized Tuner, she/he only need to:


**1) Inherit a tuner of a base Tuner class**
```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
```

**2) Implement receive trial result function**
```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
    
    def receive_trial_result(self, parameter_id, parameters, reward):
    '''
    Record an observation of the objective function
    '''
    # you code implements here.
    ...
```
**3) Implement generate parameter function**
```python
from nni.tuner import Tuner

class CustomizedTuner(Tuner):
    def __init__(self, ...):
        ...
    
    def receive_trial_result(self, parameter_id, parameters, reward):
    '''
    Record an observation of the objective function
    parameter_id: int
    parameters: object created by 'generate_parameters()'
    reward: object reported by trial
    '''
    # your code implements here.
    ...
    
    def generate_parameters(self, parameter_id):
    '''
    Returns a set of trial (hyper-)parameters, as a serializable object
    parameter_id: int
    '''
    # your code implements here.
    ...
```
**4) Write a script to run Tuner**
```python
import argparse

import CustomizedTuner

def main():
    parser = argparse.ArgumentParser(description='parse command line parameters.')
    # parse your tuner arg here.
    ...
    FLAGS, unparsed = parser.parse_known_args()

    tuner = CustomizedTuner(...)
    tuner.run()

main()
```

Please noted in **2)** and **3)**. The parameter configures from ```generate_parameters``` function, will be package as json object by nni SDK. And nni SDK will unpack json object so the Trial will receive the exact same configure from Tuner.

User could override the ```run``` function in ```CustomizedTuner``` class, which could help user to control the process logic in Tuner, such as control handle request from Trial.

```receive_trial_result``` will receive ```the parameter_id, parameters, reward``` as parameters input. Also, Tuner will receive the ```reward``` object are exactly same reward that Trial send.

More detail example you could see:
> * [evlution-tuner](https://msrasrg.visualstudio.com/NeuralNetworkIntelligenceOpenSource/_git/Default?path=%2Fsrc%2Fsdk%2Fpynni%2Fnni%2Fevolution_tuner&version=GBmaster)
> * [hyperopt-tuner](https://msrasrg.visualstudio.com/NeuralNetworkIntelligenceOpenSource/_git/Default?path=%2Fsrc%2Fsdk%2Fpynni%2Fnni%2Fhyperopt_tuner&version=GBmaster)
> * [ga-customer-tuner](https://msrasrg.visualstudio.com/NeuralNetworkIntelligenceOpenSource/_git/Default?path=%2Fexamples%2Ftuners%2Fga_customer_tuner&version=GBmaster)