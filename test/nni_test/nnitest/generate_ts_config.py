# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import os
import glob
import argparse
from utils import get_yml_content, dump_yml_content

TRAINING_SERVICE_FILE = os.path.join('config', 'training_service.yml')
TRAINING_SERVICE_FILE_V2 = os.path.join('config', 'training_service_v2.yml')

def update_training_service_config(args):
    config = get_yml_content(TRAINING_SERVICE_FILE)
    if args.nni_manager_ip is not None and args.config_version == 'v1':
        config[args.ts]['nniManagerIp'] = args.nni_manager_ip
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
        if args.debug is not None:
            config[args.ts]['debug'] = args.debug.lower() == 'true'
    elif args.ts == 'kubeflow' and args.reuse_mode == 'False':
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
        config[args.ts]['kubeflowConfig']['reuse'] = False
    elif args.ts == 'kubeflow' and args.reuse_mode == 'True':
        config = get_yml_content(TRAINING_SERVICE_FILE_V2)
        config[args.ts]['trainingService']['worker']['dockerImage'] = args.nni_docker_image
        config[args.ts]['trainingService']['storage']['azureAccount'] = args.azs_account
        config[args.ts]['trainingService']['storage']['azureShare'] = args.azs_share
        config[args.ts]['trainingService']['storage']['keyVaultName'] = args.keyvault_vaultname
        config[args.ts]['trainingService']['storage']['keyVaultKey'] = args.keyvault_name
        config[args.ts]['nni_manager_ip'] = args.nni_manager_ip
        dump_yml_content(TRAINING_SERVICE_FILE_V2, config)
    elif args.ts == 'frameworkcontroller' and args.reuse_mode == 'False':
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
        config[args.ts]['frameworkcontrollerConfig']['reuse'] = False
    elif args.ts == 'frameworkcontroller' and args.reuse_mode == 'True':
        config = get_yml_content(TRAINING_SERVICE_FILE_V2)
        config[args.ts]['trainingService']['taskRoles'][0]['dockerImage'] = args.nni_docker_image
        config[args.ts]['trainingService']['storage']['azureAccount'] = args.azs_account
        config[args.ts]['trainingService']['storage']['azureShare'] = args.azs_share
        config[args.ts]['trainingService']['storage']['keyVaultName'] = args.keyvault_vaultname
        config[args.ts]['trainingService']['storage']['keyVaultKey'] = args.keyvault_name
        config[args.ts]['nni_manager_ip'] = args.nni_manager_ip
        dump_yml_content(TRAINING_SERVICE_FILE_V2, config)
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
        if args.azurestoragetoken is not None:
            config[args.ts]['sharedStorage']['storageAccountKey'] = args.azurestoragetoken
        if args.nfs_server is not None:
            config[args.ts]['sharedStorage']['nfsServer'] = args.nfs_server
        if args.local_mount_point is not None:
            config[args.ts]['sharedStorage']['localMountPoint'] = args.local_mount_point
        if args.remote_mount_point is not None:
            config[args.ts]['sharedStorage']['remoteMountPoint'] = args.remote_mount_point
        if args.exported_directory is not None:
            config[args.ts]['sharedStorage']['exportedDirectory'] = args.exported_directory
    elif args.ts == 'adl':
        if args.nni_docker_image is not None:
            config[args.ts]['trial']['image'] = args.nni_docker_image
        if args.checkpoint_storage_class is not None:
            config[args.ts]['trial']['checkpoint']['storageClass'] = args.checkpoint_storage_class
        if args.checkpoint_storage_size is not None:
            config[args.ts]['trial']['checkpoint']['storageSize'] = args.checkpoint_storage_size
        if args.adaptive is not None:
            config[args.ts]['trial']['adaptive'] = args.adaptive
        if args.adl_nfs_server is not None and args.adl_nfs_path is not None and args.adl_nfs_container_mount_path is not None:
            # default keys in nfs is empty, need to initialize
            config[args.ts]['trial']['nfs'] = {}
            config[args.ts]['trial']['nfs']['server'] = args.adl_nfs_server
            config[args.ts]['trial']['nfs']['path'] = args.adl_nfs_path
            config[args.ts]['trial']['nfs']['container_mount_path'] = args.nadl_fs_container_mount_path
    elif args.ts == 'aml':
        if args.nni_docker_image is not None:
            config[args.ts]['trial']['image'] = args.nni_docker_image
        if args.subscription_id is not None:
            config[args.ts]['amlConfig']['subscriptionId'] = args.subscription_id
        if args.resource_group is not None:
            config[args.ts]['amlConfig']['resourceGroup'] = args.resource_group
        if args.workspace_name is not None:
            config[args.ts]['amlConfig']['workspaceName'] = args.workspace_name
        if args.compute_target is not None:
            config[args.ts]['amlConfig']['computeTarget'] = args.compute_target
    dump_yml_content(TRAINING_SERVICE_FILE, config)

    if args.ts == 'hybrid':
        config = get_yml_content(TRAINING_SERVICE_FILE_V2)
        config[args.ts]['trainingService'][0]['machineList'][0]['user'] = args.remote_user
        config[args.ts]['trainingService'][0]['machineList'][0]['host'] = args.remote_host
        config[args.ts]['trainingService'][0]['machineList'][0]['password'] = args.remote_pwd
        config[args.ts]['trainingService'][0]['machineList'][0]['port'] = args.remote_port
        config[args.ts]['trainingService'][2]['subscriptionId'] = args.subscription_id
        config[args.ts]['trainingService'][2]['resourceGroup'] = args.resource_group
        config[args.ts]['trainingService'][2]['workspaceName'] = args.workspace_name
        config[args.ts]['trainingService'][2]['computeTarget'] = args.compute_target
        config[args.ts]['nni_manager_ip'] = args.nni_manager_ip
        dump_yml_content(TRAINING_SERVICE_FILE_V2, config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=str, choices=['pai', 'kubeflow', 'remote', 'local', 'frameworkcontroller', 'adl', 'aml', 'hybrid'], default='pai')
    parser.add_argument("--config_version", type=str, choices=['v1', 'v2'], default='v1')
    parser.add_argument("--nni_docker_image", type=str)
    parser.add_argument("--nni_manager_ip", type=str)
    parser.add_argument("--reuse_mode", type=str, default='False')
    # args for remote with shared storage
    parser.add_argument("--azurestoragetoken", type=str)
    parser.add_argument("--nfs_server", type=str)
    parser.add_argument("--local_mount_point", type=str)
    parser.add_argument("--remote_mount_point", type=str)
    parser.add_argument("--exported_directory", type=str)
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
    parser.add_argument("--debug", type=str)
    # args for kubeflow and frameworkController
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
    # args for adl
    parser.add_argument("--checkpoint_storage_class", type=str)
    parser.add_argument("--checkpoint_storage_size", type=str)
    parser.add_argument("--adaptive", type=str)
    parser.add_argument("--adl_nfs_server", type=str)
    parser.add_argument("--adl_nfs_path", type=str)
    parser.add_argument("--adl_nfs_container_mount_path", type=str)
    # args for aml
    parser.add_argument("--subscription_id", type=str)
    parser.add_argument("--resource_group", type=str)
    parser.add_argument("--workspace_name", type=str)
    parser.add_argument("--compute_target", type=str)
    args = parser.parse_args()

    update_training_service_config(args)
