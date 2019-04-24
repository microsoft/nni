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


import time
import requests
from .url_utils import check_status_url
from .constants import REST_TIME_OUT
from .common_utils import print_error

def rest_put(url, data, timeout, show_error=False):
    '''Call rest put method'''
    try:
        response = requests.put(url, headers={'Accept': 'application/json', 'Content-Type': 'application/json'},\
                                data=data, timeout=timeout)
        return response
    except Exception as exception:
        if show_error:
            print_error(exception)
        return None

def rest_post(url, data, timeout, show_error=False):
    '''Call rest post method'''
    try:
        response = requests.post(url, headers={'Accept': 'application/json', 'Content-Type': 'application/json'},\
                                 data=data, timeout=timeout)
        return response
    except Exception as exception:
        if show_error:
            print_error(exception)
        return None

def rest_get(url, timeout, show_error=False):
    '''Call rest get method'''
    try:
        response = requests.get(url, timeout=timeout)
        return response
    except Exception as exception:
        if show_error:
            print_error(exception)
        return None

def rest_delete(url, timeout, show_error=False):
    '''Call rest delete method'''
    try:
        response = requests.delete(url, timeout=timeout)
        return response
    except Exception as exception:
        if show_error:
            print_error(exception)
        return None

def check_rest_server(rest_port):
    '''Check if restful server is ready'''
    retry_count = 5
    for _ in range(retry_count):
        response = rest_get(check_status_url(rest_port), REST_TIME_OUT)
        if response:
            if response.status_code == 200:
                return True, response
            else:
                return False, response
        else:
            time.sleep(3)
    return  False, response

def check_rest_server_quick(rest_port):
    '''Check if restful server is ready, only check once'''
    response = rest_get(check_status_url(rest_port), 5)
    if response and response.status_code == 200:
        return True, response
    return False, None

def check_response(response):
    '''Check if a response is success according to status_code'''
    if response and response.status_code == 200:
        return True
    return False
