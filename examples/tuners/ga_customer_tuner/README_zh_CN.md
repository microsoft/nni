# 如何使用 ga_customer_tuner?

This tuner is a customized tuner which only suitable for trial whose code path is "~/nni/examples/trials/ga_squad", type `cd ~/nni/examples/trials/ga_squad` and check readme.md to get more information for ga_squad trial.

# 配置

如果要在 Experiment 中使用 ga_customer_tuner 可按照下列格式来配置：

    tuner:
      codeDir: ~/nni/examples/tuners/ga_customer_tuner
      classFileName: customer_tuner.py
      className: CustomerTuner
      classArgs:
        optimize_mode: maximize