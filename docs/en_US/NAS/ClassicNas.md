# Classic NAS Algorithms

In classic NAS algorithms, each architecture is trained as a trial and the NAS algorithm acts as a tuner. Thus, this training mode naturally fits within the NNI hyper-parameter tuning framework, where Tuner generates new architecture for the next trial and trials run in the training service.

The following example shows how to use classic NAS algorithms. You can see it is quite similar to NNI hyper-parameter tuning.

```python
model = Net()

# get the chosen architecture from tuner and apply it on model
get_and_apply_next_architecture(model)
train(model)  # your code for training the model
acc = test(model)  # test the trained model
nni.report_final_result(acc)  # report the performance of the chosen architecture
```

First, instantiate the model. Search space has been defined in this model through `LayerChoice` and `InputChoice`. After that, user should invoke `get_and_apply_next_architecture(model)` to settle down to a specific architecture. This function receives the architecture from tuner (i.e., the classic NAS algorithm) and applies the architecture to `model`. At this point, `model` becomes a specific architecture rather than a search space. Then users are free to train this model just like training a normal PyTorch model. After get the accuracy of this model, users should invoke `nni.report_final_result(acc)` to report the result to the tuner.

The search space should be generated and sent to Tuner. As with the NNI NAS API, the search space is embedded in the user code. Users can use "[nnictl ss_gen](../Tutorial/Nnictl.md)" to generate the search space file. Then put the path of the generated search space in the field `searchSpacePath` of `config.yml`. The other fields in `config.yml` can be filled by referring [this tutorial](../Tutorial/QuickStart.md).

You can use the [NNI tuners](../Tuner/BuiltinTuner.md) to do the search. Currently, only PPO Tuner supports NAS search spaces.

We support a standalone mode for easy debugging, where you can directly run the trial command without launching an NNI experiment. This is for checking whether your trial code can correctly run. The first candidate(s) are chosen for `LayerChoice` and `InputChoice` in this standalone mode.

A complete example can be found [here](https://github.com/microsoft/nni/tree/master/examples/nas/classic_nas/config_nas.yml).