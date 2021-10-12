import os


# TODO: need to be refactored to support automatic download

DATABASE_DIR = os.environ.get("NASBENCHMARK_DIR", os.path.expanduser("~/.nni/nasbenchmark"))
