Medianstop Assessor on NNI
===

## Median Stop

Medianstop is a simple early stopping rule mentioned in the [paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46180.pdf). It stops a pending trial X at step S if the trial’s best objective value by step S is strictly worse than the median value of the running averages of all completed trials’ objectives reported up to step S.