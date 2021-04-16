import json
from pathlib import Path
import signal
from nni.experiment import *

search_space = json.load(open('search_space.json'))

exp = Experiment('remote')

machine = RemoteMachineConfig()
machine.host = '127.0.0.1'
machine.user = 'lz'
machine.password = 'cffbk'
#machine.ssh_key_file = '/home/lz/.ssh/id_rsa'
exp.config.training_service.machine_list = [machine]
exp.config.training_service.reuse_mode = True
#exp.config.training_service[0].use_active_gpu = True

exp.config.trial_concurrency = 2
exp.config.max_trial_number = 5
exp.config.search_space = search_space
exp.config.trial_command = 'python3 mnist.py'
exp.config.trial_gpu_number = 0
exp.config.nni_manager_ip = '127.0.0.1'
exp.config.tuner = AlgorithmConfig(
    name = 'TPE',
    class_args = {'optimize_mode': 'maximize'}
)

#exp.config.code_directory = Path(__file__).parent
#print(exp.config.code_directory)

exp.start(8002, debug=True)
signal.pause()
