# 自动调优的 Python API 参考

```eval_rst
.. contents::
```

## Trial

```eval_rst
..  autofunction:: nni.get_next_parameter
..  autofunction:: nni.get_current_parameter
..  autofunction:: nni.report_intermediate_result
..  autofunction:: nni.report_final_result
..  autofunction:: nni.get_experiment_id
..  autofunction:: nni.get_trial_id
..  autofunction:: nni.get_sequence_id
```

## Tuner

```eval_rst
..  autoclass:: nni.tuner.Tuner
    :members:

..  autoclass:: nni.hyperopt_tuner.hyperopt_tuner.HyperoptTuner
    :members:

..  autoclass:: nni.evolution_tuner.evolution_tuner.EvolutionTuner
    :members:

..  autoclass:: nni.smac_tuner.SMACTuner
    :members:

..  autoclass:: nni.gridsearch_tuner.GridSearchTuner
    :members:

..  autoclass:: nni.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismTuner
    :members:

..  autoclass:: nni.metis_tuner.metis_tuner.MetisTuner
    :members:

..  autoclass:: nni.ppo_tuner.PPOTuner
    :members:

..  autoclass:: nni.batch_tuner.batch_tuner.BatchTuner
    :members:

..  autoclass:: nni.gp_tuner.gp_tuner.GPTuner
    :members:
```

## Assessor

```eval_rst
..  autoclass:: nni.assessor.Assessor
    :members:

..  autoclass:: nni.assessor.AssessResult
    :members:

..  autoclass:: nni.curvefitting_assessor.CurvefittingAssessor
    :members:

..  autoclass:: nni.medianstop_assessor.MedianstopAssessor
    :members:
```

## Advisor

```eval_rst
..  autoclass:: nni.msg_dispatcher_base.MsgDispatcherBase
    :members:

..  autoclass:: nni.hyperband_advisor.hyperband_advisor.Hyperband
    :members:

..  autoclass:: nni.bohb_advisor.bohb_advisor.BOHB
    :members:
```

## 工具

```eval_rst
..  autofunction:: nni.utils.merge_parameter
```
