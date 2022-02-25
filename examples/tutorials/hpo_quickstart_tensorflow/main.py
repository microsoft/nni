"""
NNI HPO Quickstart with TensorFlow
==================================
This tutorial optimizes the model in `official TensorFlow quickstart` with auto-tuning.

The tutorial consists of 3 steps: 

 1. Modify the model for auto-tuning.
 2. Define hyperparameters' search space.
 3. Configure the experiment.
 4. Start experiment.

_official TensorFlow quickstart: https://www.tensorflow.org/tutorials/quickstart/beginner
"""

from pathlib import Path

# %%
# Step 1: Prepare the model
# -------------------------
# In first step, you need to prepare the model to be tuned.
#
# The model should be put in a separate script,
# because it will be evaluated many times concurrently,
# and possibly even trained on distributed platforms.
#
# In this tutorial, the model is defined in this script (FIXME:link).
#
# Please understand the model before continue to next step.

# %%
# Step 2: Define search space
# ---------------------------
# In model code, we have prepared 4 hyperparameters to be tuned:
# *dense_units*, *activation_type*, *dropout_rate*, and *learning_rate*.
#
# Assuming we have following prior knowledge for these hyperparameters:
#
#  1. *dense_units* should be one of 64, 128, 256.
#  2. *activation_type* should be one of 'relu', 'tanh', 'swish', or None.
#  3. *dropout_rate* should be a float between 0.5 and 0.9.
#  4. *learning_rate* should be a float between 0.0001 and 0.1, and it follows exponential distribution.
#
# Then similar to ``numpy.random``,
# in NNI we call *dense_units* and *activation_type* ``choice``s;
# *dropout_rate* is ``uniform``;
# and *learning_rate* is ``loguniform``.
#
# So we define the search space as follow:
#
# (For full specification of search space, check the reference. (FIXME:link))

search_space = {
    'dense_units': {'_type': 'choice', '_value': [64, 128, 256]},
    'activation_type': {'_type': 'choice', '_value': ['relu', 'tanh', 'swish', None]},
    'dropout_rate': {'_type': 'uniform', '_value': [0.5, 0.9]},
    'learning_rate': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
}

# %%
# Step 3: Configure the experiment
# --------------------------------
# The *experiment config* specifies:
#  1. Model code (called *trial code*)
#  2. Search space
#  3. Tuning algorithm
#  4. How to run models
#
# In the tutorial, we run the models in _local_ mode.
# This means the models will be trained on current machine, and will not involve any distributed service.
#
# For full specification of experiment config, check the reference. (FIXME:link)
from nni.experiment import Experiment
experiment = Experiment('local')

# %%
# Specify the model.
#
# In NNI, evaluation of each hyperparameter set is called a *trial*.
# So the model script is called *trial code*.
#
# This part is a little bit sophisticated here due to underlying implementation of NNI tutorials.
# For common usage, it can much simpler:
#
# .. code-block::
#
#     experiment.config.trial_command = 'python model.py'
#     experiment.config.trial_code_directory = '.'  # relative to PWD
experiment.config.trial_command = sys.executable + ' model.py'  # FIXME: support nodebook
experiment.config.trial_code_directory = Path(__file__).parent

# %%
# Specify the search space defined above.
experiment.config.search_space = search_space

# %%
# Specify tuning algorithm.
#
# Here we use TPE tuner. (FIXME:link)
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# %%
# Evaluate 10 sets of hyperparameters in total, and concurrently evaluate 4 sets at a time.
#
# Please note that ``max_trial_number`` is set to a small number here for a quick example.
# In real world usecases it is suggested to run at least 50 trials with default TPE config,
# because the algorithm takes 20 trials to warm up. 
#
# Alternatively you can leave ``max_trial_number`` unset,
# and it will keep running trials until manually stopped.
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 4

# %%
# Step 4: Run the experiment
# --------------------------
experiment.start(8080)

# %%
# Now you can open ``https://localhost:8080`` in a browser to view the web portal.

# %%
# Stop experiment
# ---------------
# When you do not need the experiment anymore, you can call ``experiment.stop()`` to clean up.

# Or alternatively you can press Ctrl-C and wait for some seconds. It will also clean up properly.
