import os
import sys
import subprocess
import yaml

def pre_run(user_cmd):
    # run the python script to generate nni.yaml
    new_env = os.environ.copy()
    new_env['RETIARII_PREPARE'] = 'retiarii_prepare'
    p = subprocess.Popen(['python3']+user_cmd, env=new_env)
    p.wait()
    # generate command for prerun by advisor
    user_cmd[0] = os.path.join(os.getcwd(), user_cmd[0])
    prerun_cmd = ' '.join(['python3'] + user_cmd)
    # update nni.yaml
    with open('nni.yaml') as fp:
        nni_config = yaml.load(fp, Loader=yaml.FullLoader)
    nni_config['advisor'] = {
        'codeDir': 'sdk/nni_integration',
        'classFileName': 'advisor_entry.py',
        'className': 'NasAdvisor',
        'classArgs': {
            'command': prerun_cmd
        }
    }
    nni_config['trial'] = {
        'codeDir': '.',
        'command': 'python3 -m sdk.nni_integration.trial_main',
        'gpuNum': 1
    }
    with open('nni.yaml', 'w') as fp:
        yaml.dump(nni_config, fp)

def start_nni_experiment():
    cmd = 'nnictl create --config nni.yaml'
    subprocess.Popen(cmd.split(' ')).wait()

if __name__ == '__main__':
    print(type(sys.argv), sys.argv)
    user_cmd = sys.argv[1:]
    pre_run(user_cmd)
    start_nni_experiment()
