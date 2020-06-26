from nni.nas.benchmark.nasbench101 import query_nb101_computed_stats
from nni.nas.benchmark.nasbench201 import query_nb201_computed_stats
from nni.nas.benchmark.nds import query_nds_computed_stats

print(next(query_nb101_computed_stats(None, None)))
print(next(query_nb201_computed_stats(None, None, None)))
print(next(query_nds_computed_stats('residual_bottleneck', None, None, None, None, None)))
print(next(query_nds_computed_stats('residual_basic', None, None, None, None, None)))
print(next(query_nds_computed_stats('vanilla', None, None, None, None, None)))
print(next(query_nds_computed_stats('nas_cell', None, None, None, None, None)))

s = set()
t = set()
for run in query_nds_computed_stats(None, None, None, None, None, None):
    proposer_model_family = (run['config']['proposer'], run['config']['model_family'])
    if proposer_model_family not in t:
        t.add(proposer_model_family)
        if run['config']['model_family'] != 'nas_cell':
            print(run['config'])
    assert run['config']['model_spec'].get('bot_muls', [1])[0] in [0, 1]
    assert run['config']['model_spec'].get('ds', [1])[0] == 1
    assert run['config']['model_spec'].get('ss', [1])[0] in [1, 4, 8], run['config']
    assert run['config']['model_spec'].get('ws', [16])[0] in [16, 32, 64], run['config']['model_spec']
    assert run['config']['model_spec'].get('num_gs', [1])[0] == 1
    if run['config']['model_family'] == 'nas_cell':
        for v in run['config']['cell_spec'].values():
            if isinstance(v, str):
                if v not in s:
                    s.add(v)
                    print(v)
print(t)