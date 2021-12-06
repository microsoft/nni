import os


ENV_NASBENCHMARK_DIR = 'NASBENCHMARK_DIR'
ENV_NNI_HOME = 'NNI_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'


def _get_nasbenchmark_dir():
    nni_home = os.path.expanduser(
        os.getenv(ENV_NNI_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'nni')))
    return os.getenv(ENV_NASBENCHMARK_DIR, os.path.join(nni_home, 'nasbenchmark'))


DATABASE_DIR = _get_nasbenchmark_dir()

DB_URLS = {
    'nasbench101': 'https://nni.blob.core.windows.net/nasbenchmark/nasbench101-209f5694.db',
    'nasbench201': 'https://nni.blob.core.windows.net/nasbenchmark/nasbench201-b2b60732.db',
    'nds': 'https://nni.blob.core.windows.net/nasbenchmark/nds-5745c235.db'
}
