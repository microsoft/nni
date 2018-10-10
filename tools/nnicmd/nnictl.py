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
from .launcher import create_experiment, resume_experiment
from .updater import update_searchspace, update_concurrency, update_duration
from .nnictl_utils import *
from .package_management import *

def nni_help_info(*args):
    print('please run "nnictl {positional argument} --help" to see nnictl guidance')

def parse_args():
    '''Definite the arguments users need to follow and input'''
    parser = argparse.ArgumentParser(prog='nnictl', description='use nnictl command to control nni experiments')
    parser.set_defaults(func=nni_help_info)

    # create subparsers for args with sub values
    subparsers = parser.add_subparsers()

    # parse start command
    parser_start = subparsers.add_parser('create', help='create a new experiment')
    parser_start.add_argument('--config', '-c', required=True, dest='config', help='the path of yaml config file')
    parser_start.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_start.set_defaults(func=create_experiment)

    # parse resume command
    parser_resume = subparsers.add_parser('resume', help='resume a new experiment')
    parser_resume.add_argument('--experiment', '-e', dest='id', help='ID of the experiment you want to resume')
    parser_resume.add_argument('--manager', '-m', default='nnimanager', dest='manager')
    parser_resume.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_resume.set_defaults(func=resume_experiment)

    # parse update command
    parser_updater = subparsers.add_parser('update', help='update the experiment')
    #add subparsers for parser_updater
    parser_updater_subparsers = parser_updater.add_subparsers()
    parser_updater_searchspace = parser_updater_subparsers.add_parser('searchspace', help='update searchspace')
    parser_updater_searchspace.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_updater_searchspace.add_argument('--filename', '-f', required=True)
    parser_updater_searchspace.set_defaults(func=update_searchspace)
    parser_updater_concurrency = parser_updater_subparsers.add_parser('concurrency', help='update concurrency')
    parser_updater_concurrency.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_updater_concurrency.add_argument('--value', '-v', required=True)
    parser_updater_concurrency.set_defaults(func=update_concurrency)
    parser_updater_duration = parser_updater_subparsers.add_parser('duration', help='update duration')
    parser_updater_duration.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_updater_duration.add_argument('--value', '-v', required=True)
    parser_updater_duration.set_defaults(func=update_duration)

    #parse stop command
    parser_stop = subparsers.add_parser('stop', help='stop the experiment')
    parser_stop.add_argument('--port', '-p', required=True, dest='port', help='the port of restful server')
    parser_stop.set_defaults(func=stop_experiment)

    #parse trial command
    parser_trial = subparsers.add_parser('trial', help='get trial information')
    #add subparsers for parser_trial
    parser_trial_subparsers = parser_trial.add_subparsers()
    parser_trial_ls = parser_trial_subparsers.add_parser('ls', help='list trial jobs')
    parser_trial_ls.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_trial_ls.set_defaults(func=trial_ls)
    parser_trial_kill = parser_trial_subparsers.add_parser('kill', help='kill trial jobs')
    parser_trial_kill.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_trial_kill.add_argument('--trialid', '-t', required=True, dest='trialid', help='the id of trial to be killed')
    parser_trial_kill.set_defaults(func=trial_kill)

    #parse experiment command
    parser_experiment = subparsers.add_parser('experiment', help='get experiment information')
    #add subparsers for parser_experiment
    parser_experiment_subparsers = parser_experiment.add_subparsers()
    parser_experiment_show = parser_experiment_subparsers.add_parser('show', help='show the information of experiment')
    parser_experiment_show.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_experiment_show.set_defaults(func=list_experiment)
    parser_experiment_status = parser_experiment_subparsers.add_parser('status', help='show the status of experiment')
    parser_experiment_status.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_experiment_status.set_defaults(func=experiment_status)

    #parse config command
    parser_config = subparsers.add_parser('config', help='get config information')
    parser_config_subparsers = parser_config.add_subparsers()
    parser_config_show = parser_config_subparsers.add_parser('show', help='show the information of config')
    parser_config_show.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_config_show.set_defaults(func=get_config)

    #parse log command
    parser_log = subparsers.add_parser('log', help='get log information')
    # add subparsers for parser_log
    parser_log_subparsers = parser_log.add_subparsers()
    parser_log_stdout = parser_log_subparsers.add_parser('stdout', help='get stdout information')
    parser_log_stdout.add_argument('--port', default=51188, dest='port', help='the port of restful server')
    parser_log_stdout.add_argument('--tail', '-T', dest='tail', type=int, help='get tail -100 content of stdout')
    parser_log_stdout.add_argument('--head', '-H', dest='head', type=int, help='get head -100 content of stdout')
    parser_log_stdout.add_argument('--path', action='store_true', default=False, help='get the path of stdout file')
    parser_log_stdout.set_defaults(func=log_stdout)
    parser_log_stderr = parser_log_subparsers.add_parser('stderr', help='get stderr information')
    parser_log_stderr.add_argument('--port', default=51188, dest='port', help='the port of restful server')
    parser_log_stderr.add_argument('--tail', '-T', dest='tail', type=int, help='get tail -100 content of stderr')
    parser_log_stderr.add_argument('--head', '-H', dest='head', type=int, help='get head -100 content of stderr')
    parser_log_stderr.add_argument('--path', action='store_true', default=False, help='get the path of stderr file')
    parser_log_stderr.set_defaults(func=log_stderr)
    parser_log_trial = parser_log_subparsers.add_parser('trial', help='get trial log path')
    parser_log_trial.add_argument('--port', '-p', default=51188, dest='port', help='the port of restful server')
    parser_log_trial.add_argument('--id', '-I', dest='id', help='find trial log path by id')
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


    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    parse_args()
