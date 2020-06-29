import os
import sys
import time
import json
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
    parser.add_argument('--script', help='script name')
    parser.add_argument('--node_count', help='the nodeCount of a run in aml')
    args = parser.parse_args()

    ws = Workspace(args.subscription_id, args.resource_group, args.workspace_name)
    compute_target = ComputeTarget(workspace=ws, name=args.computer_target)
    experiment = Experiment(ws, args.experiment_name)

    run_config = RunConfiguration()
    run_config.environment.docker.enabled = True
    run_config.environment.docker.base_image = args.docker_image
    run_config.target = compute_target
    run_config.node_count = args.node_count
    config = ScriptRunConfig(source_directory=args.code_dir, script=args.script, run_config=run_config)
    run = experiment.submit(config)
    print(run.get_details()["runId"])
    while True:
        line = sys.stdin.readline().rstrip()
        if line == 'update_status':
            print('status:' + run.get_status())
        elif line == 'tracking_url':
            print('tracking_url:' + run.get_portal_url())
        elif line == 'stop':
            run.cancel()
            exit(0)
        elif line == 'receive':
            print('receive:' + json.dumps(run.get_metrics()))
        elif line:
            items = line.split(':')
            if items[0] == 'command':
                run.log('nni_manager', line[8:])