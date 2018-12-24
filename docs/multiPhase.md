**Create multi-phase experiment**

Multi-phase experiments refer to experiments whose trial jobs request multiple hyper parameters from tuner and report multiple final results to NNI.
To use multi-phase experiment, the Tuner needs to be a subclass of nni.multi_phase.MultiPhaseTuner. For example, this [ENAS tuner](https://github.com/countif/enas_nni/blob/master/nni/examples/tuners/enas/src/nni_controller.py) is a multi-phase Tuner.
```multiPhase``` field needs to be set to ```true``` in configuration file, for example:

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


Then, nni.get_next_parameter() API can be invoked for multiple times in a trial, for example:

```python
for i in range(5):
    # get parameters form tuner
    hyper_params = nni.get_next_parameter()
    
    # consume the params
    # ...
```

API ```nni.report_final_result()``` can be called one time for each hyper parameters.
