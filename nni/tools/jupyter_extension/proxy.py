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
        ports = _get_experiment_ports()
        if not ports:
            self.set_status(404)
            return

        if path == 'index':
            if len(ports) > 1:  # if there is more than one running experiments, show experiment list
                self.redirect('experiment')
            else:  # if there is only one running experiment, show that experiment
                self.redirect('oview')
            return

        r = requests.get(f'http://localhost:{ports[0]}/{path}')
        self.set_status(r.status_code)
        for key, value in r.headers.items():
            self.add_header(key, value)
        self.finish(r.content)

    # TODO: post, put, etc

    def set_default_headers(self):
        self.clear_header('Content-Type')
        self.clear_header('Date')

def _get_experiment_ports():
    experiment_list_path = Path.home() / 'nni-experiments/.experiment'
    if not experiment_list_path.exists():
        return None
    experiments = json.load(open(experiment_list_path))
    return [exp['port'] for exp in experiments.values() if exp['status'] != 'STOPPED']
