# How to use ga_customer_tuner?
This tuner is a customized tuner which only suitable for trial whose code path is "~/nni/examples/trials/ga_squad",
type `cd ~/nni/examples/trials/ga_squad` and check readme.md to get more information for ga_squad trial.

# config
If you want to use ga_customer_tuner in your experiment, you could set config file as following format:

```
tuner:
  codeDir: ~/nni/examples/tuners/ga_customer_tuner
  classFileName: customer_tuner.py
  className: CustomerTuner
  classArgs:
    optimize_mode: maximize
```
