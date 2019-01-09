**Write a Trial Run on NNI**
===

A **Trial** in NNI is an individual attempt at applying a set of parameters on a model. 

To define a NNI trial, you need to firstly define the set of parameters and then update the model. NNI provide two approaches for you to define a trial: `NNI API` and `NNI Python annotation`.

## NNI API

>Step 1 - Prepare a SearchSpace parameters file. 

An example is shown below:

```json
{
    "dropout_rate":{"_type":"uniform","_value":[0.1,0.5]},
    "conv_size":{"_type":"choice","_value":[2,3,5,7]},
    "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
    "learning_rate":{"_type":"uniform","_value":[0.0001, 0.1]}
}
```

Refer to [SearchSpaceSpec.md](./SearchSpaceSpec.md) to learn more about search space.

>Step 2 - Update model codes

~~~~
2.1 Declare NNI API
    Include `import nni` in your trial code to use NNI APIs. 

2.2 Get predefined parameters
    Use the following code snippet: 

        RECEIVED_PARAMS = nni.get_next_parameter()

    to get hyper-parameters' values assigned by tuner. `RECEIVED_PARAMS` is an object, for example: 

        {"conv_size": 2, "hidden_size": 124, "learning_rate": 0.0307, "dropout_rate": 0.2029}

2.3 Report NNI results
    Use the API:

        `nni.report_intermediate_result(accuracy)`

    to send `accuracy` to assessor.

    Use the API:

        `nni.report_final_result(accuracy)`

    to send `accuracy` to tuner.
~~~~

**NOTE**:

~~~~
accuracy - The `accuracy` could be any python object, but  if you use NNI built-in tuner/assessor, `accuracy` should be a numerical variable (e.g. float, int).
assessor - The assessor will decide which trial should early stop based on the history performance of trial (intermediate result of one trial).
tuner    - The tuner will generate next parameters/architecture based on the explore history (final result of all trials).
~~~~

>Step 3 - Enable NNI API

To enable NNI API mode, you need to set useAnnotation to *false* and provide the path of SearchSpace file (you just defined in step 1):

```
useAnnotation: false
searchSpacePath: /path/to/your/search_space.json
```

You can refer to [here](./ExperimentConfig.md) for more information about how to set up experiment configurations.

You can refer to [here](../examples/trials/README.md) for more information about how to write trial code using NNI APIs.

## NNI Python Annotation

An alternative to write a trial is to use NNI's syntax for python. Simple as any annotation, NNI annotation is working like comments in your codes. You don't have to make structure or any other big changes to your existing codes. With a few lines of NNI annotation, you will be able to:
* annotate the variables you want to tune 
* specify in which range you want to tune the variables
* annotate which variable you want to report as intermediate result to `assessor`
* annotate which variable you want to report as the final result (e.g. model accuracy) to `tuner`.

Again, take MNIST as an example, it only requires 2 steps to write a trial with NNI Annotation.

>Step 1 - Update codes with annotations 

Please refer the following tensorflow code snippet for NNI Annotation, the highlighted 4 lines are annotations that help you to: (1) tune batch\_size and (2) dropout\_rate, (3) report test\_acc every 100 steps, and (4) at last report test\_acc as final result.

>What noteworthy is: as these new added codes are annotations, it does not actually change your previous codes logic, you can still run your code as usual in environments without NNI installed.

```diff
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
+   """@nni.variable(nni.choice(50, 250, 500), name=batch_size)"""
    batch_size = 128
    for i in range(10000):
        batch = mnist.train.next_batch(batch_size)
+       """@nni.variable(nni.choice(1, 5), name=dropout_rate)"""
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

>NOTE
>>`@nni.variable` will take effect on its following line
>>
>>`@nni.report_intermediate_result`/`@nni.report_final_result` will send the data to assessor/tuner at that line. 
>>
>>Please refer to [Annotation README](../tools/nni_annotation/README.md) for more information about annotation syntax and its usage.


>Step 2 - Enable NNI Annotation
In the yaml configure file, you need to set *useAnnotation* to true to enable NNI annotation:

```yaml
useAnnotation: true
```

## More Trial Example

* [Automatic Model Architecture Search for Reading Comprehension.](../examples/trials/ga_squad/README.md)
