import logging
from typing import Any, Optional

import requests

_logger = logging.getLogger(__name__)

timeout = 20

def request(method: str, port: Optional[int], api: str, data: Any = None, prefix: Optional[str] = None) -> Any:
    if port is None:
        raise RuntimeError('Experiment is not running')

    url_parts = [
        f'http://localhost:{port}',
        prefix,
        'api/v1/nni',
        api
    ]
    url = '/'.join(part.strip('/') for part in url_parts if part)

    if data is None:
        resp = requests.request(method, url, timeout=timeout)
    else:
        resp = requests.request(method, url, json=data, timeout=timeout)

    if not resp.ok:
        _logger.error('rest request %s %s failed: %s %s', method.upper(), url, resp.status_code, resp.text)
    resp.raise_for_status()

    if method.lower() in ['get', 'post'] and len(resp.content) > 0:
        return resp.json()

def get(port: Optional[int], api: str, prefix: Optional[str] = None) -> Any:
    return request('get', port, api, prefix=prefix)

def post(port: Optional[int], api: str, data: Any, prefix: Optional[str] = None) -> Any:
    return request('post', port, api, data, prefix=prefix)

def put(port: Optional[int], api: str, data: Any, prefix: Optional[str] = None) -> None:
    request('put', port, api, data, prefix=prefix)

def delete(port: Optional[int], api: str, prefix: Optional[str] = None) -> None:
    request('delete', port, api, prefix=prefix)
