# Write a Trial Run on NNI

A **Trial** in NNI is an individual attempt at applying a configuration (e.g., a set of hyper-parameters) on a model.

To define an NNI trial, you need to firstly define the set of parameters (i.e., search space) and then update the model. NNI provide two approaches for you to define a trial: [NNI API](#nni-api) and [NNI Python annotation](#nni-annotation). You could also refer to [here](#more-examples) for more trial examples.

<a name="nni-api"></a>
## NNI API

### Step 1 - Prepare a SearchSpace parameters file. 

An example is shown below:

```json
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}
```

Refer to [SearchSpaceSpec.md](./SearchSpaceSpec.md) to learn more about search space. Tuner will generate configurations from this search space, that is, choosing a value for each hyperparameter from the range.

### Step 2 - Update model codes

- Import NNI

    Include `import nni` in your trial code to use NNI APIs. 

- Get configuration from Tuner
    
```python
RECEIVED_PARAMS = nni.get_next_parameter()
```
`RECEIVED_PARAMS` is an object, for example: 
`{"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}`.

- Report metric data periodically (optional)

```python
nni.report_intermediate_result(metrics)
```
`metrics` could be any python object. If users use NNI built-in tuner/assessor, `metrics` can only have two formats: 1) a number e.g., float, int, 2) a dict object that has a key named `default` whose value is a number. This `metrics` is reported to [assessor](BuiltinAssessors.md). Usually, `metrics` could be periodically evaluated loss or accuracy.

- Report performance of the configuration

```python
nni.report_final_result(metrics)
```
`metrics` also could be any python object. If users use NNI built-in tuner/assessor, `metrics` follows the same format rule as that in `report_intermediate_result`, the number indicates the model's performance, for example, the model's accuracy, loss etc. This `metrics` is reported to [tuner](BuiltinTuner.md).

### Step 3 - Enable NNI API

To enable NNI API mode, you need to set useAnnotation to *false* and provide the path of SearchSpace file (you just defined in step 1):

```yaml
useAnnotation: false
searchSpacePath: /path/to/your/search_space.json
```

You can refer to [here](ExperimentConfig.md) for more information about how to set up experiment configurations.

*Please refer to [here](https://nni.readthedocs.io/en/latest/sdk_reference.html) for more APIs (e.g., `nni.get_sequence_id()`) provided by NNI.


<a name="nni-annotation"></a>
## NNI Python Annotation

An alternative to writing a trial is to use NNI's syntax for python. Simple as any annotation, NNI annotation is working like comments in your codes. You don't have to make structure or any other big changes to your existing codes. With a few lines of NNI annotation, you will be able to:

* annotate the variables you want to tune 
* specify in which range you want to tune the variables
* annotate which variable you want to report as intermediate result to `assessor`
* annotate which variable you want to report as the final result (e.g. model accuracy) to `tuner`. 

Again, take MNIST as an example, it only requires 2 steps to write a trial with NNI Annotation.

### Step 1 - Update codes with annotations 

The following is a tensorflow code snippet for NNI Annotation, where the highlighted four lines are annotations that help you to: 
  1. tune batch\_size and dropout\_rate
  2. report test\_acc every 100 steps
  3. at last report test\_acc as final result.

What noteworthy is: as these newly added codes are annotations, it does not actually change your previous codes logic, you can still run your code as usual in environments without NNI installed.

```diff
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
+   """@nni.variable(nni.choice(50, 250, 500), name=batch_size)"""
    batch_size = 128
    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
+       """@nni.variable(nni.choice(0.1, 0.5), name=dropout_rate)"""
        dropout_rate = 0.5
        mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                mnist_network.labels: batch[1],
                                                mnist_network.keep_prob: dropout_rate})
        if i % 100 == 0:
            test_acc = mnist_network.accuracy.eval(
                feed_dict={mnist_network.images: mnist.test.images,
                            mnist_network.labels: mnist.test.labels,
                            mnist_network.keep_prob: 1.0})
+           """@nni.report_intermediate_result(test_acc)"""

    test_acc = mnist_network.accuracy.eval(
        feed_dict={mnist_network.images: mnist.test.images,
                    mnist_network.labels: mnist.test.labels,
                    mnist_network.keep_prob: 1.0})
+   """@nni.report_final_result(test_acc)"""
```

**NOTE**: 
- `@nni.variable` will take effect on its following line, which is an assignment statement whose leftvalue must be specified by the keyword `name` in `@nni.variable`.
- `@nni.report_intermediate_result`/`@nni.report_final_result` will send the data to assessor/tuner at that line. 

For more information about annotation syntax and its usage, please refer to [Annotation](AnnotationSpec.md). 


### Step 2 - Enable NNI Annotation

In the YAML configure file, you need to set *useAnnotation* to true to enable NNI annotation:
```
useAnnotation: true
```


## Where are my trials?

### Local Mode

In NNI, every trial has a dedicated directory for them to output their own data. In each trial, an environment variable called `NNI_OUTPUT_DIR` is exported. Under this directory, you could find each trial's code, data and other possible log. In addition, each trial's log (including stdout) will be re-directed to a file named `trial.log` under that directory.

If NNI Annotation is used, trial's converted code is in another temporary directory. You can check that in a file named `run.sh` under the directory indicated by `NNI_OUTPUT_DIR`. The second line (i.e., the `cd` command) of this file will change directory to the actual directory where code is located. Below is an example of `run.sh`:
```shell
#!/bin/bash
cd /tmp/user_name/nni/annotation/tmpzj0h72x6 #This is the actual directory
export NNI_PLATFORM=local
export NNI_SYS_DIR=/home/user_name/nni/experiments/$experiment_id$/trials/$trial_id$
export NNI_TRIAL_JOB_ID=nrbb2
export NNI_OUTPUT_DIR=/home/user_name/nni/experiments/$eperiment_id$/trials/$trial_id$
export NNI_TRIAL_SEQ_ID=1
export MULTI_PHASE=false
export CUDA_VISIBLE_DEVICES=
eval python3 mnist.py 2>/home/user_name/nni/experiments/$experiment_id$/trials/$trial_id$/stderr
echo $? `date +%s000` >/home/user_name/nni/experiments/$experiment_id$/trials/$trial_id$/.nni/state
```

### Other Modes

When runing trials on other platform like remote machine or PAI, the environment variable `NNI_OUTPUT_DIR` only refers to the output directory of the trial, while trial code and `run.sh` might not be there. However, the `trial.log` will be transmitted back to local machine in trial's directory, which defaults to `~/nni/experiments/$experiment_id$/trials/$trial_id$/`

For more information, please refer to [HowToDebug](HowToDebug.md)

<a name="more-examples"></a>
## More Trial Examples

* [MNIST examples](MnistExamples.md)
* [Finding out best optimizer for Cifar10 classification](Cifar10Examples.md)
* [How to tune Scikit-learn on NNI](SklearnExamples.md)
* [Automatic Model Architecture Search for Reading Comprehension.](SquadEvolutionExamples.md)
* [Tuning GBDT on NNI](GbdtExample.md)
