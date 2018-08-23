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


import psutil
from socket import AddressFamily
from .rest_utils import rest_get
from .config_utils import Config
from subprocess import Popen, PIPE
from .common_utils import print_error, print_normal
from .constants import STDOUT_FULL_PATH, STDERR_FULL_PATH

def start_web_ui(port):
    '''start web ui'''
    cmds = ['serve', '-s', '-n', '$HOME/.nni/webui', '-l', str(port)]
    stdout_file = open(STDOUT_FULL_PATH, 'a+')
    stderr_file = open(STDERR_FULL_PATH, 'a+')
    webui_process = Popen(cmds, stdout=stdout_file, stderr=stderr_file)
    if webui_process.returncode is None:
        webui_url_list = []
        for name, info in psutil.net_if_addrs().items():
            for addr in info:
                if AddressFamily.AF_INET == addr.family:
                    webui_url_list.append('http://{}:{}'.format(addr.address, port))
        nni_config = Config()
        nni_config.set_config('webuiUrl', webui_url_list)
    else:
        print_error('Failed to start webui')
    return webui_process

def stop_web_ui():
    '''stop web ui'''
    nni_config = Config()
    webuiPid = nni_config.get_config('webuiPid')
    if not webuiPid:
        return False
    #detect webui process first
    try:
        parent_process = psutil.Process(webuiPid)
        if not parent_process or not parent_process.is_running():
            return False
    except:
        return False
    #then kill webui process
    try:
        #in some environment, there will be multi processes, kill them all
        parent_process = psutil.Process(webuiPid)
        child_process_list = parent_process.children(recursive=True)
        for child_process in child_process_list:
            if child_process.is_running():
                child_process.kill()
        if parent_process.is_running():
            parent_process.kill()
        return True
    except Exception as e:
        print_error(e)
        return False

def check_web_ui():
    '''check if web ui is alive'''
    nni_config = Config()
    url_list = nni_config.get_config('webuiUrl')
    if not url_list:
        return False
    for url in url_list:
        response = rest_get(url, 20)
        if response and response.status_code == 200:
            return True
    return False