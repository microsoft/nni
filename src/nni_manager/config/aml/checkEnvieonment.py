import os
import time
from argparse import ArgumentParser
from azureml.core import Experiment, RunConfiguration, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.core.run import RUNNING_STATES, RunStatus, Run
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--subscription_id', help='the subscription id of aml')
    parser.add_argument('--resource_group', help='the resource group of aml')
    parser.add_argument('--workspace_name', help='the workspace name of aml')
    parser.add_argument('--experiment_name', help='the experiment name')
    parser.add_argument('--environment_id', help='the experiment id')
    args = parser.parse_args()

    ws = Workspace(args.subscription_id, args.resource_group, args.workspace_name)
    experiment = Experiment(ws, args.experiment_name)

    run_list = experiment.get_runs()
    for run in run_list:
        if run['runId'] == args.environment_id:
            print(run['status'])
            return
    print('Unknown')
