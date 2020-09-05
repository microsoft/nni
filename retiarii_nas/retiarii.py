import os
import shutil
import sys
import subprocess
import yaml
from examples.allstars.trainers.darts import example as darts_example
from examples.allstars.trainers.proxylessnas import example as proxyless_g_example

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
    if user_cmd[1] == 'textnas_e2e':
        nni_config['advisor'] = {
            'codeDir': 'sdk/nni_integration',
            'classFileName': 'advisor_entry.py',
            'className': 'NasAdvisor',
            'gpuIndices': 2,
            'classArgs': {
                'command': prerun_cmd
            }
        }
    else:
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
    cmd = 'nnictl create --config nni.yaml --port 9090'
    subprocess.Popen(cmd.split(' ')).wait()

if __name__ == '__main__':
    try:
        os.remove('experiment.yaml')
    except FileNotFoundError:
        pass
    finished = False
    if sys.argv[1].lower() == '-l':
        print('Launch the script with:\npython retiarii.py e2e_launch.py <nas_mode>\n' + 
              'nas_mode can be: nasnet_e2e, mnasnet_e2e, amoebanet_e2e, proxylessnas_e2e, ' +
              'nasbench101_e2e, nasbench201_e2e, fbnet_e2e, textnas_e2e, chamnet_e2e, ' +
              'onceforall_e2e, singlepathnas_e2e, spos, hierarchical, wann, path_level, ' +
              'darts, proxyless_g.')
        sys.exit(0)
    if len(sys.argv) > 2:
        if sys.argv[2] in ['hierarchical', 'wann', 'path_level']:
            algorithm = sys.argv[2]
            port = sys.argv[3] if len(sys.argv) > 3 else '8080'
            shutil.copyfile(f'examples/combo/{algorithm}/experiment.yaml', 'experiment.yaml')
            subprocess.run(f'nnictl create --config sdk/nni_integration/nni.yaml --port {port}'.split())
            finished = True
        elif sys.argv[2] in ['darts', 'proxyless_g']:
            if sys.argv[2] == 'darts':
                darts_example()
            elif sys.argv[2] == 'proxyless_g':
                proxyless_g_example()
            finished = True
    if not finished:
        user_cmd = sys.argv[1:]
        pre_run(user_cmd)
        start_nni_experiment()
