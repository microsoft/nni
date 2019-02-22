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
        config[args.ts]['paiConfig']['userName'] = args.pai_user
        config[args.ts]['paiConfig']['passWord'] = args.pai_pwd
        config[args.ts]['paiConfig']['host'] = args.pai_host
        config[args.ts]['trial']['image'] = args.nni_docker_image
        config[args.ts]['trial']['dataDir'] = args.data_dir
        config[args.ts]['trial']['outputDir'] = args.output_dir
    elif args.ts == 'kubeflow':
        config[args.ts]['kubeflowConfig']['nfs']['server'] = args.nfs_server
        config[args.ts]['kubeflowConfig']['nfs']['path'] = args.nfs_path

    dump_yml_content(TRAINING_SERVICE_FILE, config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=str, choices=['pai', 'kubeflow'], default='pai')
    # args for PAI
    parser.add_argument("--pai_user", type=str)
    parser.add_argument("--pai_pwd", type=str)
    parser.add_argument("--pai_host", type=str)
    parser.add_argument("--nni_docker_image", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    # args for kubeflow
    parser.add_argument("--nfs_server", type=str)
    parser.add_argument("--nfs_path", type=str)
    args = parser.parse_args()

    update_training_service_config(args)
