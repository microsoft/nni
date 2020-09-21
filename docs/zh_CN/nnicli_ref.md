# NNI 客户端

NNI client 是 `nnictl` 的python API，提供了对常用命令的实现。 相比于命令行，用户可以通过此 API 来在 python 代码中直接操控实验，收集实验结果并基于实验结果进行更加高级的分析。 示例如下：

```
from nnicli import Experiment

# 创建 Experiment 实例
exp = Experiment() 

# 启动 Experiment，并将实例连接到该 Experiment
# 也可以使用 `resume_experiment`, `view_experiment` 或 `connect_experiment`
# 同一实例中只有上面中的一个函数应该被调用
exp.start_experiment('nni/examples/trials/mnist-pytorch/config.yml', port=9090)

# 更新 Experiment 的并发设置
exp.update_concurrency(3)

# 获取 Experiment 的信息
print(exp.get_experiment_status())
print(exp.get_job_statistics())
print(exp.list_trial_jobs())

# 关闭 Experiment，并将实例与 Experiment 解除关联
exp.stop_experiment()
```

## 参考

```eval_rst
..  autoclass:: nnicli.Experiment
    :members:
..  autoclass:: nnicli.TrialJob
    :members:
..  autoclass:: nnicli.TrialHyperParameters
    :members:
..  autoclass:: nnicli.TrialMetricData
    :members:
..  autoclass:: nnicli.TrialResult
    :members:
```
