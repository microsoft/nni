# How to install this customized tuner as a builtin tuner

## Prepare package installation source
Run following command to prepare the installation source:
```bash
python setup.py develop
```

## Install this tuner as a builtin tuner
Run following command to install it as a builtin tuner:
```
nnictl package install ./
```

Then run command `nnictl package list`, you should be able to see that demotuner is installed:
```
+-----------------+------------+-----------+--------=-------------+------------------------------------------+
|      Name       |    Type    | Installed |      Class Name      |               Module Name                |
+-----------------+------------+-----------+----------------------+------------------------------------------+
| demotuner       | tuners     | Yes       | DemoTuner            | demo_tuner                               |
+-----------------+------------+-----------+----------------------+------------------------------------------+
```

## Use the installed tuner in experiment
Now you can use the demotuner in experiment configuration file the same way as other builtin tuners:

```yaml
tuner:
  builtinTunerName: demotuner
  classArgs:
    #choice: maximize, minimize
    optimize_mode: maximize
```
