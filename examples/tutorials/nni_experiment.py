"""
Start and Manage a New Experiment
=================================
"""

# %%
# Configure Search Space
# ----------------------

search_space = {
    "C": {"_type": "quniform", "_value": [0.1, 1, 0.1]},
    "kernel": {"_type": "choice", "_value": ["linear", "rbf", "poly", "sigmoid"]},
    "degree": {"_type": "choice", "_value": [1, 2, 3, 4]},
    "gamma": {"_type": "quniform", "_value": [0.01, 0.1, 0.01]},
    "coef0": {"_type": "quniform", "_value": [0.01, 0.1, 0.01]}
}

# %%
# Configure Experiment
# --------------------

from nni.experiment import Experiment
experiment = Experiment('local')
experiment.config.experiment_name = 'Example'
experiment.config.trial_concurrency = 2
experiment.config.max_trial_number = 10
experiment.config.search_space = search_space
experiment.config.trial_command = 'python scripts/trial_sklearn.py'
experiment.config.trial_code_directory = './'
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
experiment.config.training_service.use_active_gpu = True

# %%
# Start Experiment
# ----------------
experiment.start(8080)

# %%
# Experiment View & Control
# -------------------------
#
# View the status of experiment.
experiment.get_status()

# %%
# Wait until at least one trial finishes.
import time

for _ in range(10):
    stats = experiment.get_job_statistics()
    if any(stat['trialJobStatus'] == 'SUCCEEDED' for stat in stats):
        break
    time.sleep(10)

# %%
# Export the experiment data.
experiment.export_data()

# %%
# Get metric of jobs
experiment.get_job_metrics()

# %%
# Stop Experiment
# ---------------
experiment.stop()
