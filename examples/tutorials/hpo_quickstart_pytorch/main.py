"""
NNI HPO Quickstart with PyTorch
===============================
This tutorial optimizes the model in `official PyTorch quickstart`_ with auto-tuning.

The tutorial consists of 4 steps: 

 1. Modify the model for auto-tuning.
 2. Define hyperparameters' search space.
 3. Configure the experiment.
 4. Run the experiment.

.. _official PyTorch quickstart: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

# %%
# Step 1: Prepare the model
# -------------------------
# In first step, you need to prepare the model to be tuned.
#
# The model should be put in a separate script.
# It will be evaluated many times concurrently,
# and possibly will be trained on distributed platforms.
#
# In this tutorial, the model is defined in :doc:`model.py <model>`.
#
# Please understand the model code before continue to next step.

# %%
# Step 2: Define search space
# ---------------------------
# In model code, we have prepared 3 hyperparameters to be tuned:
# *features*, *lr*, and *momentum*.
#
# Here we need to define their *search space* so the tuning algorithm can sample them in desired range.
#
# Assuming we have following prior knowledge for these hyperparameters:
#
#  1. *features* should be one of 128, 256, 512, 1024.
#  2. *lr* should be a float between 0.0001 and 0.1, and it follows exponential distribution.
#  3. *momentum* should be a float between 0 and 1.
#
# In NNI, the space of *features* is called ``choice``;
# the space of *lr* is called ``loguniform``;
# and the space of *momentum* is called ``uniform``.
# You may have noticed, these names are derived from ``numpy.random``.
#
# For full specification of search space, check :doc:`the reference </hpo/search_space>`.
#
# Now we can define the search space as follow:

search_space = {
    'features': {'_type': 'choice', '_value': [128, 256, 512, 1024]},
    'lr': {'_type': 'loguniform', '_value': [0.0001, 0.1]},
    'momentum': {'_type': 'uniform', '_value': [0, 1]},
}

# %%
# Step 3: Configure the experiment
# --------------------------------
# NNI uses an *experiment* to manage the HPO process.
# The *experiment config* defines how to train the models and how to explore the search space.
# 
# In this tutorial we use a *local* mode experiment,
# which means models will be trained on local machine, without using any special training platform.
from nni.experiment import Experiment
experiment = Experiment('local')

# %%
# Now we start to configure the experiment.
#
# Firstly, specify the model code.
# In NNI evaluation of each hyperparameter set is called a *trial*.
# So the model script is called *trial code*.
#
# If you are using Linux system without Conda, you many need to change ``python`` to ``python3``.
#
# When ``trial_code_directory`` is a relative path, it relates to current working directory.
# To run ``main.py`` from a different path, you can set trial code directory to ``Path(__file__).parent``.
experiment.config.trial_command = 'python model.py'
experiment.config.trial_code_directory = '.'

# %%
# Then specify the search space we defined above:
experiment.config.search_space = search_space

# %%
# Choose a tuning algorithm.
# Here we use :doc:`TPE tuner </hpo/tuners>`.
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'

# %%
# Specify how many trials to run.
# Here we evaluate 10 sets of hyperparameters in total, and concurrently evaluate 4 sets at a time.
#
# Please note that ``max_trial_number`` here is merely for a quick example.
# With default config TPE tuner requires 20 trials to warm up.
# In real world max trial number is commonly set to 100+.
#
# You can also set ``max_experiment_duration = '1h'`` to limit running time.
#
# And alternatively, you can skip this part and set no limit at all.
# The experiment will run forever until you press Ctrl-C.
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 4

# %%
# Step 4: Run the experiment
# --------------------------
# Now the experiment is ready. Choose a port and launch it.
#
# You can use the web portal to view experiment status: http://localhost:8080.
experiment.run(8080)
