import logging
from typing import Any, Optional

import requests

_logger = logging.getLogger(__name__)

url_template = 'http://localhost:{}/api/v1/nni{}'
timeout = 20

def request(method: str, port: Optional[int], api: str, data: Any = None) -> Any:
    if port is None:
        raise RuntimeError('Experiment is not running')
    url = url_template.format(port, api)
    if data is None:
        resp = requests.request(method, url, timeout=timeout)
    else:
        resp = requests.request(method, url, json=data, timeout=timeout)
    if not resp.ok:
        _logger.error('rest request %s %s failed: %s %s', method.upper(), url, resp.status_code, resp.text)
    resp.raise_for_status()
    if method.lower() in ['get', 'post'] and len(resp.content) > 0:
        return resp.json()

def get(port: Optional[int], api: str) -> Any:
    return request('get', port, api)

def post(port: Optional[int], api: str, data: Any) -> Any:
    return request('post', port, api, data)

def put(port: Optional[int], api: str, data: Any) -> None:
    request('put', port, api, data)

def delete(port: Optional[int], api: str) -> None:
    request('delete', port, api)
