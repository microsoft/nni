# 如何使用 ga_customer_tuner?

此定制的 Tuner 仅适用于代码 "~/nni/examples/trials/ga_squad"，输入 `cd ~/nni/examples/trials/ga_squad` 查看 readme.md 来了解 ga_squad 的更多信息。

# 配置

如果要在 Experiment 中使用 ga_customer_tuner 可按照下列格式来配置：

    tuner:
      codeDir: ~/nni/examples/tuners/ga_customer_tuner
      classFileName: customer_tuner.py
      className: CustomerTuner
      classArgs:
        optimize_mode: maximize