import logging
from typing import Any

import requests

_logger = logging.getLogger(__name__)

url_template = 'http://localhost:{}/api/v1/nni{}'
timeout = 20

def get(port: int, api: str) -> Any:
    url = url_template.format(port, api)
    resp = requests.get(url, timeout=timeout)
    if not resp.ok:
        _logger.error('rest request GET %s %s failed: %s %s', port, api, resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()

def post(port: int, api: str, data: Any) -> Any:
    url = url_template.format(port, api)
    resp = requests.post(url, json=data, timeout=timeout)
    if not resp.ok:
        _logger.error('rest request POST %s %s failed: %s %s', port, api, resp.status_code, resp.text)
    resp.raise_for_status()
    return resp.json()

def put(port: int, api: str, data: Any) -> None:
    url = url_template.format(port, api)
    resp = requests.put(url, json=data, timeout=timeout)
    if not resp.ok:
        _logger.error('rest request PUT %s %s failed: %s', port, api, resp.status_code)
    resp.raise_for_status()
