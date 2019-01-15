**Create multi-phase experiment**

Typically each trial job get one single hyper parameter from tuner and do some kind of experiment, let's say train a model with that hyper parameter and reports its result to tuner.  Sometimes you may want to share information between your models. However each trial job is in a seperate process and may cross operating systems, it is not convenient to share information between your models. In such case, you can leverage NNI multi-phase experiment to share information between models easily.

Multi-phase experiments refer to experiments whose trial jobs request multiple hyper parameters from tuner and report multiple final results to NNI.
To use multi-phase experiment, please follow below steps:

1.  Implement nni.multi_phase.MultiPhaseTuner. For example, this [ENAS tuner](https://github.com/countif/enas_nni/blob/master/nni/examples/tuners/enas/src/nni_controller_ptb.py) is a multi-phase Tuner which implements nni.multi_phase.MultiPhaseTuner.

2. Set ```multiPhase``` field to ```true```, and configure your tuner implemented in step 1 as customized tuner in configuration file, for example:

```yml
...
multiPhase: true
tuner:
  codeDir: tuners/enas
  classFileName: nni_controller_ptb.py
  className: ENASTuner
  classArgs:
    say_hello: "hello"
...
```


3. Invoke nni.get_next_parameter() API for multiple times as needed in a trial, for example:

```python
for i in range(5):
    # get parameter from tuner
    tuner_param = nni.get_next_parameter()
    
    # consume the params
    # ...
    # report final result somewhere for the parameter retrieved above
    nni.report_final_result()
    # ...
```
