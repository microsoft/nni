import functools
import json


json_dumps = functools.partial(json.dumps, sort_keys=True)
