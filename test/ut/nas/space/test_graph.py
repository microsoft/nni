import json
from pathlib import Path
from nni.nas.space import GraphModelSpace


json_files = [
    'mnist_tensorflow.json'
]


def test_model_load_dump():
    for json_file in json_files:
        path = Path(__file__).parent / json_file
        _test_file(path)


def _test_file(json_path):
    orig_ir = json.load(json_path.open())
    model = GraphModelSpace._load(_internal=True, **orig_ir)
    dump_ir = model._dump()

    # add default values to JSON, so we can compare with `==`
    for graph in orig_ir.values():
        if isinstance(graph, dict):
            if 'inputs' not in graph:
                graph['inputs'] = None
            if 'outputs' not in graph:
                graph['outputs'] = None

    # debug output
    #json.dump(orig_ir, open('_orig.json', 'w'), indent=4)
    #json.dump(dump_ir, open('_dump.json', 'w'), indent=4)

    # skip model id and mutators
    dump_ir.pop('model_id')
    dump_ir.pop('_mutators')

    assert orig_ir == dump_ir
