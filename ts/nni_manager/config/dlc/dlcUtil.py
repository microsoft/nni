# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import logging
import os
import pathlib
import sys
import traceback
from argparse import ArgumentParser
# ref: https://help.aliyun.com/document_detail/203290.html?spm=a2c4g.11186623.6.727.6f9b5db6bzJh4x
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi.models import Config
from alibabacloud_pai_dlc20201203.models import * #CreateJobRequest, JobSpec

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--type', help='the type of job spec')
    parser.add_argument('--image', help='the docker image of job')
    parser.add_argument('--job_type', choices=['TFJob', 'PyTorchJob'], help='the job type')
    parser.add_argument('--pod_count', type=int, default=1, help='pod count')
    parser.add_argument('--ecs_spec', help='ecs spec')
    parser.add_argument('--region', help='region')
    parser.add_argument('--workspace_id', help='workspace id for your project')
    parser.add_argument('--nas_data_source_id', help='nas data_source_id of DLC dataset configuration')
    parser.add_argument('--oss_data_source_id', help='oss data_source_id of DLC dataset configuration')
    parser.add_argument('--access_key_id', help='access_key_id')
    parser.add_argument('--access_key_secret', help='access_key_secret')
    parser.add_argument('--experiment_name', help='the experiment name')
    parser.add_argument('--user_command', help='user command')
    parser.add_argument('--log_dir', help='exception log dir')
    args = parser.parse_args()

    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=os.path.join(args.log_dir, 'dlc_exception.log'),
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)

    # DLC submit
    try:

        # init client
        client = Client(
            Config(
                access_key_id=args.access_key_id,
                access_key_secret=args.access_key_secret,
                region_id=args.region,
                endpoint=f'pai-dlc.{args.region}.aliyuncs.com'
            )
        )

        nas_1 = DataSourceItem(
            data_source_type='nas',
            data_source_id=args.nas_data_source_id,
        )

        oss = None
        if args.oss_data_source_id:
            oss = DataSourceItem(
                data_source_type='oss',
                data_source_id=args.oss_data_source_id,
            )

        # job spec
        spec = JobSpec(
            type=args.type,
            image=args.image,
            pod_count=args.pod_count,
            ecs_spec=args.ecs_spec,
        )

        data_sources = [nas_1]
        if oss:
            data_sources = [nas_1, oss]
        req = CreateJobRequest(
            display_name=args.experiment_name,
            job_type=args.job_type,
            job_specs=[spec],
            data_sources=data_sources,
            user_command=args.user_command,
            workspace_id=args.workspace_id,
        )

        response = client.create_job(req)
        job_id = response.body.job_id
        print('job id: ' + job_id)

        while True:
            line = sys.stdin.readline().rstrip()
            if line == 'update_status':
                print('status:' + client.get_job(job_id).body.status)
            elif line == 'tracking_url':
                #TODO: 1. get this url by api? 2. change this url in private dlc mode.
                print('tracking_url:' + f'https://pai-dlc.console.aliyun.com/#/jobs/detail?jobId={job_id}&regionId={args.region}')
            elif line == 'stop':
                client.stop_job(job_id)
                exit(0)
    except Exception as e:
        logging.error('DLC submit Exception: \n')
        logging.error(e, exc_info=1)
