import argparse
from utils import get_yml_content, dump_yml_content

TRAINING_SERVICE_FILE = 'training_service.yml'

def update_training_service_config(args):
    ts = get_yml_content(TRAINING_SERVICE_FILE)
    ts['pai']['paiConfig']['userName'] = args.pai_user
    ts['pai']['paiConfig']['passWord'] = args.pai_pwd
    ts['pai']['paiConfig']['host'] = args.pai_host
    ts['pai']['trial']['image'] = args.nni_docker_image
    ts['pai']['trial']['dataDir'] = args.data_dir
    ts['pai']['trial']['outputDir'] = args.output_dir

    dump_yml_content(TRAINING_SERVICE_FILE, ts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pai_user", type=str, required=True)
    parser.add_argument("--pai_pwd", type=str, required=True)
    parser.add_argument("--pai_host", type=str, required=True)
    parser.add_argument("--nni_docker_image", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    update_training_service_config(args)
