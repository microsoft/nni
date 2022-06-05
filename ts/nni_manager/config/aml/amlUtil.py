# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import time
import json
import warnings
from argparse import ArgumentParser
from azureml.core import Experiment, RunConfiguration, ScriptRunConfig, Workspace
from azureml.core.authentication import (
    AzureCliAuthentication, InteractiveLoginAuthentication, AuthenticationException
)
from azureml.core.compute import ComputeTarget
from azureml.core.run import RUNNING_STATES, RunStatus, Run
from azureml.core.conda_dependencies import CondaDependencies

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--subscription_id', help='the subscription id of aml')
    parser.add_argument('--resource_group', help='the resource group of aml')
    parser.add_argument('--workspace_name', help='the workspace name of aml')
    parser.add_argument('--compute_target', help='the compute cluster name of aml')
    parser.add_argument('--docker_image', help='the docker image of job')
    parser.add_argument('--experiment_name', help='the experiment name')
    parser.add_argument('--script_dir', help='script directory')
    parser.add_argument('--script_name', help='script name')
    args = parser.parse_args()

    try:
        auth = AzureCliAuthentication()
        auth.get_token()
    except AuthenticationException as e:
        warnings.warn(
            f'Azure-cli authentication failed: {e}',
            RuntimeWarning
        )
        warnings.warn('Falling back to interactive authentication.', RuntimeWarning)
        auth = InteractiveLoginAuthentication()

    ws = Workspace(args.subscription_id, args.resource_group, args.workspace_name, auth=auth)
    compute_target = ComputeTarget(workspace=ws, name=args.compute_target)
    experiment = Experiment(ws, args.experiment_name)
    run_config = RunConfiguration()
    run_config.environment.python.user_managed_dependencies = True
    run_config.environment.docker.enabled = True
    run_config.environment.docker.base_image = args.docker_image
    run_config.target = compute_target
    run_config.node_count = 1
    config = ScriptRunConfig(source_directory=args.script_dir, script=args.script_name, run_config=run_config)
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
            loop_count = 0
            status = run.get_status()
            # wait until the run is canceled
            while status != 'Canceled':
                if loop_count > 5:
                    print('stop_result:failed')
                    exit(0)
                loop_count += 1
                time.sleep(5)
                status = run.get_status()
            print('stop_result:success')
            exit(0)
        elif line == 'receive':
            print('receive:' + json.dumps(run.get_metrics()))
        elif line:
            items = line.split(':')
            if items[0] == 'command':
                run.log('nni_manager', line[8:])
