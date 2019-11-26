# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import requests

def rest_get(url, timeout):
    '''Call rest get method'''
    try:
        response = requests.get(url, timeout=timeout)
        return response
    except Exception as e:
        print('Get exception {0} when sending http get to url {1}'.format(str(e), url))
        return None

def rest_post(url, data, timeout, rethrow_exception=False):
    '''Call rest post method'''
    try:
        response = requests.post(url, headers={'Accept': 'application/json', 'Content-Type': 'application/json'},\
                                 data=data, timeout=timeout)
        return response
    except Exception as e:
        if rethrow_exception is True:
            raise
        print('Get exception {0} when sending http post to url {1}'.format(str(e), url))
        return None

def rest_put(url, data, timeout):
    '''Call rest put method'''
    try:
        response = requests.put(url, headers={'Accept': 'application/json', 'Content-Type': 'application/json'},\
                                data=data, timeout=timeout)
        return response
    except Exception as e:
        print('Get exception {0} when sending http put to url {1}'.format(str(e), url))
        return None

def rest_delete(url, timeout):
    '''Call rest delete method'''
    try:
        response = requests.delete(url, timeout=timeout)
        return response
    except Exception as e:
        print('Get exception {0} when sending http delete to url {1}'.format(str(e), url))
        return None
