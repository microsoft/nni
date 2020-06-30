# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
    retry_count = 20
    for _ in range(retry_count):
        response = rest_get(check_status_url(rest_port), REST_TIME_OUT)
        if response:
            if response.status_code == 200:
                return True, response
            else:
                return False, response
        else:
            time.sleep(1)
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
