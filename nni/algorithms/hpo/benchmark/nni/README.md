NNI Custom benchmarks:

This folder contains a customized framework with several NNI Hyperparameter Tuners, along with testing benchmarks.

**To experiment with an existing tuner:**

1. Select tuner type and architecture type in frameworks.yaml.
2. Find or define a benchmark in the "benchmarks" folder with name <benchmark>.yaml. 
3. Go to the project root folder and run the following command.

```bash
python runbenchmark.py NNI <benchmark> -u nni
```

**Note:**
 you can also copy those files in automlbenchmark user config directory `~/.config/automlbenchmark` to be able to use them without having to specify the `-u examples/custom` argument.
