# NNI Annotation 


## Overview

To improve user experience and reduce user effort, we design an annotation grammar. Using NNI annotation, users can adapt their code to NNI just by adding some standalone annotating strings, which does not affect the execution of the original code. 

Below is an example:

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```
The meaning of this example is that NNI will choose one of several values (0.1, 0.01, 0.001) to assign to the learning_rate variable. Specifically, this first line is an NNI annotation, which is a single string. Following is an assignment statement. What nni does here is to replace the right value of this assignment statement according to the information provided by the annotation line.


In this way, users could either run the python code directly or launch NNI to tune hyper-parameter in this code, without changing any codes.

## Types of Annotation:

In NNI, there are mainly four types of annotation:


### 1. Annotate variables

   `'''@nni.variable(sampling_algo, name)'''`

`@nni.variable` is used in NNI to annotate a variable.

**Arguments**

- **sampling_algo**: Sampling algorithm that specifies a search space. User should replace it with a built-in NNI sampling function whose name consists of an `nni.` identification and a search space type specified in [SearchSpaceSpec](SearchSpaceSpec.md) such as `choice` or `uniform`. 
- **name**: The name of the variable that the selected value will be assigned to. Note that this argument should be the same as the left value of the following assignment statement.

An example here is:

```python
'''@nni.variable(nni.choice(0.1, 0.01, 0.001), name=learning_rate)'''
learning_rate = 0.1
```

### 2. Annotate functions

   `'''@nni.function_choice(*functions, name)'''`

`@nni.function_choice` is used to choose one from several functions.

**Arguments**

- **\*functions**: Several functions that are waiting to be selected from. Note that it should be a complete function call with arguments. Such as `max_pool(hidden_layer, pool_size)`.
- **name**: The name of the function that will be replaced in the following assignment statement.

An example here is:

```python
"""@nni.function_choice(max_pool(hidden_layer, pool_size), avg_pool(hidden_layer, pool_size), name=max_pool)"""
h_pooling = max_pool(hidden_layer, pool_size)
```

### 3. Annotate intermediate result

   `'''@nni.report_intermediate_result(metrics)'''`

`@nni.report_intermediate_result` is used to report intermediate result, whose usage is the same as `nni.report_intermediate_result` in [Trials.md](Trials.md)

### 4. Annotate final result

   `'''@nni.report_final_result(metrics)'''`

`@nni.report_final_result` is used to report the final result of the current trial, whose usage is the same as `nni.report_final_result` in [Trials.md](Trials.md)
