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
import pkg_resources
from .launcher import create_experiment, resume_experiment
from .updater import update_searchspace, update_concurrency, update_duration, update_trialnum, import_data
from .nnictl_utils import *
from .package_management import *
from .constants import *
from .tensorboard_utils import *
from colorama import init
init(autoreset=True)

if os.environ.get('COVERAGE_PROCESS_START'):
    import coverage
    coverage.process_startup()

def nni_info(*args):
    if args[0].version:
        try:
            print(pkg_resources.get_distribution('nni').version)
        except pkg_resources.ResolutionError as err:
            print_error('Get version failed, please use `pip3 list | grep nni` to check nni version!')
    else:
        print('please run "nnictl {positional argument} --help" to see nnictl guidance')

def parse_args():
    '''Definite the arguments users need to follow and input'''
    parser = argparse.ArgumentParser(prog='nnictl', description='use nnictl command to control nni experiments')
    parser.add_argument('--version', '-v', action='store_true')
    parser.set_defaults(func=nni_info)

    # create subparsers for args with sub values
    subparsers = parser.add_subparsers()

    # parse start command
    parser_start = subparsers.add_parser('create', help='create a new experiment')
    parser_start.add_argument('--config', '-c', required=True, dest='config', help='the path of yaml config file')
    parser_start.add_argument('--port', '-p', default=DEFAULT_REST_PORT, dest='port', help='the port of restful server')
    parser_start.add_argument('--debug', '-d', action='store_true', help=' set debug mode')
    parser_start.set_defaults(func=create_experiment)

    # parse resume command
    parser_resume = subparsers.add_parser('resume', help='resume a new experiment')
    parser_resume.add_argument('id', nargs='?', help='The id of the experiment you want to resume')
    parser_resume.add_argument('--port', '-p', default=DEFAULT_REST_PORT, dest='port', help='the port of restful server')
    parser_resume.add_argument('--debug', '-d', action='store_true', help=' set debug mode')
    parser_resume.set_defaults(func=resume_experiment)

    # parse update command
    parser_updater = subparsers.add_parser('update', help='update the experiment')
    #add subparsers for parser_updater
    parser_updater_subparsers = parser_updater.add_subparsers()
    parser_updater_searchspace = parser_updater_subparsers.add_parser('searchspace', help='update searchspace')
    parser_updater_searchspace.add_argument('id', nargs='?', help='the id of experiment')
    parser_updater_searchspace.add_argument('--filename', '-f', required=True)
    parser_updater_searchspace.set_defaults(func=update_searchspace)
    parser_updater_concurrency = parser_updater_subparsers.add_parser('concurrency', help='update concurrency')
    parser_updater_concurrency.add_argument('id', nargs='?', help='the id of experiment')
    parser_updater_concurrency.add_argument('--value', '-v', required=True)
    parser_updater_concurrency.set_defaults(func=update_concurrency)
    parser_updater_duration = parser_updater_subparsers.add_parser('duration', help='update duration')
    parser_updater_duration.add_argument('id', nargs='?', help='the id of experiment')
    parser_updater_duration.add_argument('--value', '-v', required=True, help='the unit of time should in {\'s\', \'m\', \'h\', \'d\'}')
    parser_updater_duration.set_defaults(func=update_duration)
    parser_updater_trialnum = parser_updater_subparsers.add_parser('trialnum', help='update maxtrialnum')
    parser_updater_trialnum.add_argument('--id', '-i', dest='id', help='the id of experiment')
    parser_updater_trialnum.add_argument('--value', '-v', required=True)
    parser_updater_trialnum.set_defaults(func=update_trialnum)

    #parse stop command
    parser_stop = subparsers.add_parser('stop', help='stop the experiment')
    parser_stop.add_argument('id', nargs='?', help='the id of experiment, use \'all\' to stop all running experiments')
    parser_stop.set_defaults(func=stop_experiment)

    #parse trial command
    parser_trial = subparsers.add_parser('trial', help='get trial information')
    #add subparsers for parser_trial
    parser_trial_subparsers = parser_trial.add_subparsers()
    parser_trial_ls = parser_trial_subparsers.add_parser('ls', help='list trial jobs')
    parser_trial_ls.add_argument('id', nargs='?', help='the id of experiment')
    parser_trial_ls.set_defaults(func=trial_ls)
    parser_trial_kill = parser_trial_subparsers.add_parser('kill', help='kill trial jobs')
    parser_trial_kill.add_argument('id', nargs='?', help='the id of experiment')
    parser_trial_kill.add_argument('--trial_id', '-T', required=True, dest='trial_id', help='the id of trial to be killed')
    parser_trial_kill.set_defaults(func=trial_kill)

    #parse experiment command
    parser_experiment = subparsers.add_parser('experiment', help='get experiment information')
    #add subparsers for parser_experiment
    parser_experiment_subparsers = parser_experiment.add_subparsers()
    parser_experiment_show = parser_experiment_subparsers.add_parser('show', help='show the information of experiment')
    parser_experiment_show.add_argument('id', nargs='?', help='the id of experiment')
    parser_experiment_show.set_defaults(func=list_experiment)
    parser_experiment_status = parser_experiment_subparsers.add_parser('status', help='show the status of experiment')
    parser_experiment_status.add_argument('id', nargs='?', help='the id of experiment')
    parser_experiment_status.set_defaults(func=experiment_status)
    parser_experiment_list = parser_experiment_subparsers.add_parser('list', help='list all of running experiment ids')
    parser_experiment_list.add_argument('all', nargs='?', help='list all of experiments')
    parser_experiment_list.set_defaults(func=experiment_list)
    #import tuning data
    parser_import_data = parser_experiment_subparsers.add_parser('import', help='import additional data')
    parser_import_data.add_argument('id', nargs='?', help='the id of experiment')
    parser_import_data.add_argument('--filename', '-f', required=True)
    parser_import_data.set_defaults(func=import_data)
    #export trial data
    parser_trial_export = parser_experiment_subparsers.add_parser('export', help='export trial job results to csv or json')
    parser_trial_export.add_argument('id', nargs='?', help='the id of experiment')
    parser_trial_export.add_argument('--type', '-t', choices=['json', 'csv'], required=True, dest='type', help='target file type')
    parser_trial_export.add_argument('--filename', '-f', required=True, dest='path', help='target file path')
    parser_trial_export.set_defaults(func=export_trials_data)

    #TODO:finish webui function
    #parse board command
    parser_webui = subparsers.add_parser('webui', help='get web ui information')
    #add subparsers for parser_board
    parser_webui_subparsers = parser_webui.add_subparsers()
    parser_webui_url = parser_webui_subparsers.add_parser('url', help='show the url of web ui')
    parser_webui_url.add_argument('id', nargs='?', help='the id of experiment')
    parser_webui_url.set_defaults(func=webui_url)

    #parse config command
    parser_config = subparsers.add_parser('config', help='get config information')
    parser_config_subparsers = parser_config.add_subparsers()
    parser_config_show = parser_config_subparsers.add_parser('show', help='show the information of config')
    parser_config_show.add_argument('id', nargs='?', help='the id of experiment')
    parser_config_show.set_defaults(func=get_config)

    #parse log command
    parser_log = subparsers.add_parser('log', help='get log information')
    # add subparsers for parser_log
    parser_log_subparsers = parser_log.add_subparsers()
    parser_log_stdout = parser_log_subparsers.add_parser('stdout', help='get stdout information')
    parser_log_stdout.add_argument('id', nargs='?', help='the id of experiment')
    parser_log_stdout.add_argument('--tail', '-T', dest='tail', type=int, help='get tail -100 content of stdout')
    parser_log_stdout.add_argument('--head', '-H', dest='head', type=int, help='get head -100 content of stdout')
    parser_log_stdout.add_argument('--path', action='store_true', default=False, help='get the path of stdout file')
    parser_log_stdout.set_defaults(func=log_stdout)
    parser_log_stderr = parser_log_subparsers.add_parser('stderr', help='get stderr information')
    parser_log_stderr.add_argument('id', nargs='?', help='the id of experiment')
    parser_log_stderr.add_argument('--tail', '-T', dest='tail', type=int, help='get tail -100 content of stderr')
    parser_log_stderr.add_argument('--head', '-H', dest='head', type=int, help='get head -100 content of stderr')
    parser_log_stderr.add_argument('--path', action='store_true', default=False, help='get the path of stderr file')
    parser_log_stderr.set_defaults(func=log_stderr)
    parser_log_trial = parser_log_subparsers.add_parser('trial', help='get trial log path')
    parser_log_trial.add_argument('id', nargs='?', help='the id of experiment')
    parser_log_trial.add_argument('--trial_id', '-T', dest='trial_id', help='find trial log path by id')
    parser_log_trial.set_defaults(func=log_trial)

    #parse package command
    parser_package = subparsers.add_parser('package', help='control nni tuner and assessor packages')
    # add subparsers for parser_package
    parser_package_subparsers = parser_package.add_subparsers()
    parser_package_install = parser_package_subparsers.add_parser('install', help='install packages')
    parser_package_install.add_argument('--name', '-n', dest='name', help='package name to be installed')
    parser_package_install.set_defaults(func=package_install) 
    parser_package_show = parser_package_subparsers.add_parser('show', help='show the information of packages')
    parser_package_show.set_defaults(func=package_show)

    #parse tensorboard command
    parser_tensorboard = subparsers.add_parser('tensorboard', help='manage tensorboard')
    parser_tensorboard_subparsers = parser_tensorboard.add_subparsers()
    parser_tensorboard_start = parser_tensorboard_subparsers.add_parser('start', help='start tensorboard')
    parser_tensorboard_start.add_argument('id', nargs='?', help='the id of experiment')
    parser_tensorboard_start.add_argument('--trial_id', '-T', dest='trial_id', help='the id of trial')
    parser_tensorboard_start.add_argument('--port', dest='port', default=6006, help='the port to start tensorboard')
    parser_tensorboard_start.set_defaults(func=start_tensorboard)
    parser_tensorboard_start = parser_tensorboard_subparsers.add_parser('stop', help='stop tensorboard')
    parser_tensorboard_start.add_argument('id', nargs='?', help='the id of experiment')
    parser_tensorboard_start.set_defaults(func=stop_tensorboard)

    #parse top command
    parser_top = subparsers.add_parser('top', help='monitor the experiment')
    parser_top.add_argument('--time', '-t', dest='time', type=int, default=3, help='the time interval to update the experiment status, ' \
    'the unit is second')
    parser_top.set_defaults(func=monitor_experiment)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    parse_args()
