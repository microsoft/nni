Assessors
==============
In order to save our computing resources, NNI supports an early stop policy and creates **Assessor** to finish this job.

Assessor receives the intermediate result from Trial and decides whether the Trial should be killed by specific algorithm. Once the Trial experiment meets the early stop conditions(which means assessor is pessimistic about the final results), the assessor will kill the trial and the status of trial will be `"EARLY_STOPPED"`.

Here is an experimental result of MNIST after using 'Curvefitting' Assessor in 'maximize' mode, you can see that assessor successfully **early stopped** many trials with bad hyperparameters in advance. If you use assessor, we may get better hyperparameters under the same computing resources.

*Implemented code directory: config_assessor.yml <https://github.com/Microsoft/nni/blob/master/examples/trials/mnist/config_assessor.yml>*

..  image:: ../img/Assessor.png

Like Tuners, users can either use built-in Assessors, or customize an Assessor on their own. Please refer to the following tutorials for detail:

..  toctree::
    :maxdepth: 2

    Builtin Assessors<BuiltinAssessor>
    Customized Assessors<CustomizeAssessor>
