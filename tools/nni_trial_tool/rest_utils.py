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
