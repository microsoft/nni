import json
from pathlib import Path

import requests
from tornado.web import RequestHandler

def setup(server):
    base_url = server.web_app.settings['base_url']
    url_pattern = base_url.rstrip('/') + '/nni/(.*)'
    server.web_app.add_handlers('.*$', [(url_pattern, NniProxyHandler)])

class NniProxyHandler(RequestHandler):
    def get(self, path):
        port = _get_experiment_port()
        if port is None:
            self.set_status(404)
        else:
            r = requests.get(f'http://localhost:{port}/{path}')
            self.set_status(r.status_code)
            for key, value in r.headers.items():
                self.add_header(key, value)
            self.finish(r.content)

    # TODO: post, put, etc

    def set_default_headers(self):
        self.clear_header('Content-Type')
        self.clear_header('Date')

def _get_experiment_port():
    experiment_list_path = Path.home() / 'nni-experiments/.experiment'
    if not experiment_list_path.exists():
        return None
    experiments = json.load(open(experiment_list_path))
    for experiment in experiments.values():
        if experiment['status'] != 'STOPPED':
            return experiment['port']
