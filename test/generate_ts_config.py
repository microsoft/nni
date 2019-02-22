# Copyright (c) Microsoft Corporation
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
from utils import get_yml_content, dump_yml_content

TRAINING_SERVICE_FILE = 'training_service.yml'

def update_training_service_config(args):
    config = get_yml_content(TRAINING_SERVICE_FILE)

    if args.ts == 'pai':
        if args.pai_user is not None:
            config[args.ts]['paiConfig']['userName'] = args.pai_user
        if args.pai_pwd is not None:
            config[args.ts]['paiConfig']['passWord'] = args.pai_pwd
        if args.pai_host is not None:
            config[args.ts]['paiConfig']['host'] = args.pai_host
        if args.nni_docker_image is not None:
            config[args.ts]['trial']['image'] = args.nni_docker_image
        if args.data_dir is not None:
            config[args.ts]['trial']['dataDir'] = args.data_dir
        if args.output_dir is not None:
            config[args.ts]['trial']['outputDir'] = args.output_dir
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

    dump_yml_content(TRAINING_SERVICE_FILE, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=str, choices=['pai', 'kubeflow'], default='pai')
    parser.add_argument("--nni_docker_image", type=str)
    # args for PAI
    parser.add_argument("--pai_user", type=str)
    parser.add_argument("--pai_pwd", type=str)
    parser.add_argument("--pai_host", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    # args for kubeflow
    parser.add_argument("--nfs_server", type=str)
    parser.add_argument("--nfs_path", type=str)
    parser.add_argument("--keyvault_vaultname", type=str)
    parser.add_argument("--keyvault_name", type=str)
    parser.add_argument("--azs_account", type=str)
    parser.add_argument("--azs_share", type=str)
    args = parser.parse_args()

    update_training_service_config(args)
