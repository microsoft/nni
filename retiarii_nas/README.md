Usage:

    $ cp naive/experiment.yaml .
    $ nnictl create --config sdk/nni_integration/nni.yaml


    $ cp single-mutation/experiment.yaml .
    $ nnictl create --config sdk/nni_integration/nni.yaml


    $ cp examples/graphs/experiment.yaml .
    $ nnictl create --config sdk/nni_integration/nni.yaml

The logging system is not yet ready. Check `nnictl log stderr` to debug.

To use GPU or to increase concurrency, edit `nni.yaml`.
