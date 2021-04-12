import json
from pathlib import Path
from nni.experiment import *

search_space = json.load(open('search_space.json'))

exp = Experiment('remote')

machine = RemoteMachineConfig()
machine.host = '10.172.143.34'
machine.user = 'liuzhe'
machine.ssh_key_file = '/home/lz/.ssh/id_rsa'
exp.config.training_service.machine_list = [machine]
exp.config.training_service.reuse_mode = False
#exp.config.training_service[0].use_active_gpu = True

exp.config.trial_concurrency = 2
exp.config.max_trial_number = 5
exp.config.search_space = search_space
exp.config.trial_command = 'python3 mnist.py'
exp.config.trial_gpu_number = 0
exp.config.tuner = AlgorithmConfig(
    name = 'TPE',
    class_args = {'optimize_mode': 'maximize'}
)

#exp.config.code_directory = Path(__file__).parent
#print(exp.config.code_directory)

exp.run(8001, debug=True)
