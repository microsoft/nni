# Python API Reference of Auto Tune

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

..  autoclass:: nni.algorithms.hpo.hyperopt_tuner.hyperopt_tuner.HyperoptTuner
    :members:

..  autoclass:: nni.algorithms.hpo.evolution_tuner.evolution_tuner.EvolutionTuner
    :members:

..  autoclass:: nni.algorithms.hpo.smac_tuner.SMACTuner
    :members:

..  autoclass:: nni.algorithms.hpo.gridsearch_tuner.GridSearchTuner
    :members:

..  autoclass:: nni.algorithms.hpo.networkmorphism_tuner.networkmorphism_tuner.NetworkMorphismTuner
    :members:

..  autoclass:: nni.algorithms.hpo.metis_tuner.metis_tuner.MetisTuner
    :members:

..  autoclass:: nni.algorithms.hpo.ppo_tuner.PPOTuner
    :members:

..  autoclass:: nni.algorithms.hpo.batch_tuner.batch_tuner.BatchTuner
    :members:

..  autoclass:: nni.algorithms.hpo.gp_tuner.gp_tuner.GPTuner
    :members:
```

## Assessor

```eval_rst
..  autoclass:: nni.assessor.Assessor
    :members:

..  autoclass:: nni.assessor.AssessResult
    :members:

..  autoclass:: nni.algorithms.hpo.curvefitting_assessor.CurvefittingAssessor
    :members:

..  autoclass:: nni.algorithms.hpo.medianstop_assessor.MedianstopAssessor
    :members:
```

## Advisor

```eval_rst
..  autoclass:: nni.runtime.msg_dispatcher_base.MsgDispatcherBase
    :members:

..  autoclass:: nni.algorithms.hpo.hyperband_advisor.hyperband_advisor.Hyperband
    :members:

..  autoclass:: nni.algorithms.hpo.bohb_advisor.bohb_advisor.BOHB
    :members:
```

## Utilities

```eval_rst
..  autofunction:: nni.utils.merge_parameter
```
