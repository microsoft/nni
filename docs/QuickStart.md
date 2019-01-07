# QuickStart

## Installation

We support Linux and MacOS in current stage, Ubuntu 16.04 or higher, along with MacOS 10.14.1 are tested and supported. Simply run the following `pip install` in an environment that has `python >= 3.5`.

```bash
    python3 -m pip install --upgrade nni
```

Note:

* `--user` can be added if you want to install NNI in your home directory, which does not require any special privileges.
* If there is any error like `Segmentation fault`, please refer to [FAQ][1]
* For the `system requirements` of NNI, please refer to [Install NNI][2]

## "Hello World" example: MNIST

As long as you provide data sets, training methods and search spaces, NNI can automatically generate different sets of hyper-parameters, start corresponding training trials and find the optimal hyper-parameter configuration within the given time or number of experiments.

Here is an example script to train a CNN on MNIST dataset **without NNI**:

```python
# Please refer to source code to see the detail implementation.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''[skipped a piece of code here]'''

def main(params):
    # Import data [skipped a piece of code here]
    # Create the model [skipped a piece of code here]
    # Write log [skipped a piece of code here]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['batch_num']):
            '''[skipped a piece of code here]'''
            if i % 100 == 0:
                test_acc = mnist_network.accuracy.eval(
                    feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})
        '''[skipped a piece of code here]'''
        test_acc = mnist_network.accuracy.eval(
            feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})

if __name__ == '__main__':
    try:
        # run
        params = generate_default_params()
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
```

Note: We display the important part instead of complete file in the doc. If you want to see the full implementation, please refer to [examples/trials/mnist/mnist_without_nni.py][7]

Without NNI, user have to manually tune the configuration and start each trial. It requires high degree of expertise and a lot of experience, and NNI targets to automatically do this work. NNI will automatically tune and optimal the hyper-parameters sets by collecting the intermediate and final results matrix with the-state-of-art tuning algorithms.

If you want to use NNI to automatically train your model and find the optimal hyper-parameters, you have to do three more things:

**Three things required to do when using NNI**

1. Give a `Search Space` file in json, includes the `name` and the `distribution` (discrete valued or continuous valued) of hyper-parameters you need to search. 

```diff
+ {
+     "dropout_rate":{"_type":"uniform","_value":[0.5, 0.9]},
+     "conv_size":{"_type":"choice","_value":[2,3,5,7]},
+     "hidden_size":{"_type":"choice","_value":[124, 512, 1024]},
+     "batch_size": {"_type":"choice", "_value": [1, 4, 8, 16, 32]},
+     "learning_rate":{"_type":"choice","_value":[0.0001, 0.001, 0.01, 0.1]}
+ }
```

*Implemented code directory: [search_space.json][3]*

2. Modified your  `Trial` file to report the intermedian and final result to NNI and update your hyper-parameter sets.

```diff
  # Please refer to source code to see the detail implementation.
  import tensorflow as tf
+ import nni
  from tensorflow.examples.tutorials.mnist import input_data

  '''[skipped a piece of code here]'''

  def main(params):
      # Import data
      '''[skipped a piece of code here]'''
      # Create the model
      '''[skipped a piece of code here]'''
      # Write log
      '''[skipped a piece of code here]'''
      with tf.Session() as sess:
          sess.run(tf.global_variables_initializer())
          for i in range(params['batch_num']):
              '''[skipped a piece of code here]'''
              if i % 100 == 0:
                  test_acc = mnist_network.accuracy.eval(
                      feed_dict={mnist_network.images: mnist.test.images,
                                mnist_network.labels: mnist.test.labels,
                                mnist_network.keep_prob: 1.0})
+                 # report the imtermediate result matrix after finish each epoch
+                 nni.report_intermediate_result(test_acc)
          '''[skipped a piece of code here]'''
          test_acc = mnist_network.accuracy.eval(
              feed_dict={mnist_network.images: mnist.test.images,
                        mnist_network.labels: mnist.test.labels,
                        mnist_network.keep_prob: 1.0})
+         # report final result matrix when the trial finished all the epoch
+         nni.report_final_result(test_acc)

  if __name__ == '__main__':
      try:
          # run
+         # get parameters from tuner
+         RCV_PARAMS = nni.get_next_parameter()
          params = generate_default_params()
+         # Start the trial with the latest parameters
+         params.update(RCV_PARAMS)
          main(params)
      except Exception as exception:
          logger.exception(exception)
          raise
```

*Implemented code directory: [mnist.py][4]*

3. Define a `config` file in yaml, which declare the `path` to search space and trial, also give `other information` such as tuning algorithm, runtime and name arguments.

```diff
+ authorName: default
+ experimentName: example_mnist
+ trialConcurrency: 1
+ maxExecDuration: 1h
+ maxTrialNum: 10
+ trainingServicePlatform: local
+ # The path to Search Space
+ searchSpacePath: search_space.json
+ useAnnotation: false
+ tuner:
+   builtinTunerName: TPE
+ trial:
+   # The path and the running command of model training method(trial)
+   command: python3 mnist.py
+   codeDir: .
+   gpuNum: 0
```

*Implemented code directory: [config.yml][5]*

All the code above are stored in [examples/trials/mnist/][8]. When everything is ready, **run the config.yml file from your command line**.

```bash
    nnictl create --config nni/examples/trials/mnist/config.yml
```

Note: **nnictl** is a command line tool, which can be used to control experiments, such as start/stop/resume an experiment, start/stop NNIBoard, etc. Click [here][6] for more usage of `nnictl`

Wait for the message `INFO: Successfully started experiment!` in the command line. This message indicates that your experiment has been successfully started. And this is what we expected to get:

```
INFO: Starting restful server...
INFO: Successfully started Restful server!
INFO: Setting local config...
INFO: Successfully set local config!
INFO: Starting experiment...
INFO: Successfully started experiment!
-----------------------------------------------------------------------
The experiment id is egchD4qy
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
-----------------------------------------------------------------------

You can use these commands to get more information about the experiment
-----------------------------------------------------------------------
         commands                       description
1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
-----------------------------------------------------------------------
```

If you prepare `trial`, `search space` and `config` according to the above steps and successfully create a NNI job, NNI will automatically tune the optimal hyper-parameters and run different hyper-parameters sets for each trial according to the requirements you set. You can clearly sees its progress by NNI WebUI.

## WebUI

After you start your experiment in NNI successfully, you can find a message in the command-line interface to tell you `Web UI url` like this:

```
The Web UI urls are: http://223.255.255.1:8080   http://127.0.0.1:8080
```

Open the `Web UI url`(In this information is: `http://223.255.255.1:8080`) in your browser, you can view detail information of the experiment and all the submitted trial jobs as shown below.

### View summary page

Click the tab "Overview".

* See the experiment trial profile and search space message.
* Support to download the experiment result.

![](./img/over1.png)
* See good performance trials.

![](./img/over2.png)

### View job default metric

Click the tab "Default Metric" to see the point graph of all trials. Hover to see its specific default metric and search space message.

![](./img/accuracy.png)

### View hyper parameter

Click the tab "Hyper Parameter" to see the parallel graph.

* You can select the percentage to see top trials.
* Choose two axis to swap its positions

![](./img/hyperPara.png)

### View Trial Duration

Click the tab "Trial Duration" to see the bar graph.

![](./img/trial_duration.png)

### View trials status 

Click the tab "Trials Detail" to see the status of the all trials. Specifically:

* Trial detail: trial's id, trial's duration, start time, end time, status, accuracy and search space file.
* If you run a pai experiment, you can also see the hdfsLogPath.

![](./img/table_openrow.png)

* Kill: you can kill a job that status is running.
* Support to search for a specific trial.
* Intermediate Result Graph.

![](./img/intermediate.png)

[1]: https://github.com/Microsoft/nni/blob/master/docs/FAQ.md
[2]: https://github.com/Microsoft/nni/blob/master/docs/Installation.md
[3]: https://github.com/Microsoft/nni/blob/master/examples/trials/mnist/search_space.json
[4]: https://github.com/Microsoft/nni/blob/master/examples/trials/mnist/mnist.py
[5]: https://github.com/Microsoft/nni/blob/master/examples/trials/mnist/config.yml
[6]: https://github.com/Microsoft/nni/blob/master/docs/NNICTLDOC.md
[7]: https://github.com/Microsoft/nni/blob/43981d732fefee9d938620f2c959bf8411bd3a29/examples/trials/mnist/mnist_without_nni.py
[8]: https://github.com/Microsoft/nni/tree/master/examples/trials/mnist