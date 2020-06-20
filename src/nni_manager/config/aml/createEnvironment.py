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
    parser.add_argument('--computer_target', help='the computer cluster name of aml')
    parser.add_argument('--docker_image', help='the docker image of job')
    parser.add_argument('--experiment_name', help='the experiment name')
    parser.add_argument('--code_dir', help='code directory')
    parser.add_argument('--script', help='script')
    args = parser.parse_args()

    ws = Workspace(args.subscription_id, args.resource_group, args.workspace_name)
    compute_target = ComputeTarget(workspace=ws, name=args.computer_target)
    experiment = Experiment(ws, args.experiment_name)
    dependencies = CondaDependencies()
    dependencies.add_pip_package("azureml-sdk")
    dependencies.add_pip_package("azureml")

    run_config = RunConfiguration()
    run_config.environment.python.conda_dependencies = dependencies
    run_config.environment.docker.enabled = True
    run_config.environment.docker.base_image = args.docker_image
    run_config.target = compute_target
    run_config.node_count = 1
    config = ScriptRunConfig(source_directory=args.code_dir, script=args.script, run_config=run_config)
    script_run = experiment.submit(config)
    print(script_run.get_details()["runId"])