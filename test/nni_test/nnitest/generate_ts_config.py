# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os
import glob
import argparse
from utils import get_yml_content, dump_yml_content

TRAINING_SERVICE_FILE = os.path.join('config', 'training_service.yml')

def update_training_service_config(args):
    config = get_yml_content(TRAINING_SERVICE_FILE)
    if args.nni_manager_ip is not None:
        config[args.ts]['nniManagerIp'] = args.nni_manager_ip
    if args.ts == 'paiYarn':
        if args.pai_user is not None:
            config[args.ts]['paiYarnConfig']['userName'] = args.pai_user
        if args.pai_pwd is not None:
            config[args.ts]['paiYarnConfig']['passWord'] = args.pai_pwd
        if args.pai_host is not None:
            config[args.ts]['paiYarnConfig']['host'] = args.pai_host
        if args.nni_docker_image is not None:
            config[args.ts]['trial']['image'] = args.nni_docker_image
        if args.data_dir is not None:
            config[args.ts]['trial']['dataDir'] = args.data_dir
        if args.output_dir is not None:
            config[args.ts]['trial']['outputDir'] = args.output_dir
        if args.vc is not None:
            config[args.ts]['trial']['virtualCluster'] = args.vc
    if args.ts == 'pai':
        if args.pai_user is not None:
            config[args.ts]['paiConfig']['userName'] = args.pai_user
        if args.pai_host is not None:
            config[args.ts]['paiConfig']['host'] = args.pai_host
        if args.pai_token is not None:
            config[args.ts]['paiConfig']['token'] = args.pai_token
        if args.pai_reuse is not None:
            config[args.ts]['paiConfig']['reuse'] = args.pai_reuse.lower() == 'true'
        if args.nni_docker_image is not None:
            config[args.ts]['trial']['image'] = args.nni_docker_image
        if args.nni_manager_nfs_mount_path is not None:
            config[args.ts]['trial']['nniManagerNFSMountPath'] = args.nni_manager_nfs_mount_path
        if args.container_nfs_mount_path is not None:
            config[args.ts]['trial']['containerNFSMountPath'] = args.container_nfs_mount_path
        if args.pai_storage_config_name is not None:
            config[args.ts]['trial']['paiStorageConfigName'] = args.pai_storage_config_name
        if args.vc is not None:
            config[args.ts]['trial']['virtualCluster'] = args.vc
    elif args.ts == 'kubeflow':
        if args.nfs_server is not None:
            config[args.ts]['kubeflowConfig']['nfs']['server'] = args.nfs_server
        if args.nfs_path is not None:
            config[args.ts]['kubeflowConfig']['nfs']['path'] = args.nfs_path
        if args.keyvault_vaultname is not None:
            config[args.ts]['kubeflowConfig']['keyVault']['vaultName'] = args.keyvault_vaultname
        if args.keyvault_name is not None:
            config[args.ts]['kubeflowConfig']['keyVault']['name'] = args.keyvault_name
        if args.azs_account is not None:
            config[args.ts]['kubeflowConfig']['azureStorage']['accountName'] = args.azs_account
        if args.azs_share is not None:
            config[args.ts]['kubeflowConfig']['azureStorage']['azureShare'] = args.azs_share
        if args.nni_docker_image is not None:
            config[args.ts]['trial']['worker']['image'] = args.nni_docker_image
    elif args.ts == 'frameworkcontroller':
        if args.nfs_server is not None:
            config[args.ts]['frameworkcontrollerConfig']['nfs']['server'] = args.nfs_server
        if args.nfs_path is not None:
            config[args.ts]['frameworkcontrollerConfig']['nfs']['path'] = args.nfs_path
        if args.keyvault_vaultname is not None:
            config[args.ts]['frameworkcontrollerConfig']['keyVault']['vaultName'] = args.keyvault_vaultname
        if args.keyvault_name is not None:
            config[args.ts]['frameworkcontrollerConfig']['keyVault']['name'] = args.keyvault_name
        if args.azs_account is not None:
            config[args.ts]['frameworkcontrollerConfig']['azureStorage']['accountName'] = args.azs_account
        if args.azs_share is not None:
            config[args.ts]['frameworkcontrollerConfig']['azureStorage']['azureShare'] = args.azs_share
        if args.nni_docker_image is not None:
            config[args.ts]['trial']['taskRoles'][0]['image'] = args.nni_docker_image
    elif args.ts == 'remote':
        if args.remote_user is not None:
            config[args.ts]['machineList'][0]['username'] = args.remote_user
        if args.remote_host is not None:
            config[args.ts]['machineList'][0]['ip'] = args.remote_host
        if args.remote_port is not None:
            config[args.ts]['machineList'][0]['port'] = args.remote_port
        if args.remote_pwd is not None:
            config[args.ts]['machineList'][0]['passwd'] = args.remote_pwd
        if args.remote_reuse is not None:
            config[args.ts]['remoteConfig']['reuse'] = args.remote_reuse.lower() == 'true'

    dump_yml_content(TRAINING_SERVICE_FILE, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=str, choices=['pai', 'kubeflow', 'remote', 'local', 'frameworkcontroller'], default='pai')
    parser.add_argument("--nni_docker_image", type=str)
    parser.add_argument("--nni_manager_ip", type=str)
    # args for PAI
    parser.add_argument("--pai_user", type=str)
    parser.add_argument("--pai_pwd", type=str)
    parser.add_argument("--pai_host", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--vc", type=str)
    parser.add_argument("--pai_token", type=str)
    parser.add_argument("--pai_reuse", type=str)
    parser.add_argument("--pai_storage_config_name", type=str)
    parser.add_argument("--nni_manager_nfs_mount_path", type=str)
    parser.add_argument("--container_nfs_mount_path", type=str)
    # args for kubeflow and frameworkController
    parser.add_argument("--nfs_server", type=str)
    parser.add_argument("--nfs_path", type=str)
    parser.add_argument("--keyvault_vaultname", type=str)
    parser.add_argument("--keyvault_name", type=str)
    parser.add_argument("--azs_account", type=str)
    parser.add_argument("--azs_share", type=str)
    # args for remote
    parser.add_argument("--remote_user", type=str)
    parser.add_argument("--remote_pwd", type=str)
    parser.add_argument("--remote_host", type=str)
    parser.add_argument("--remote_port", type=int)
    parser.add_argument("--remote_reuse", type=str)
    args = parser.parse_args()

    update_training_service_config(args)
