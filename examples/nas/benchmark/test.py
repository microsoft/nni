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
for run in query_nds_computed_stats('nas_cell', None, None, None, None, None):
    for v in run['config']['cell_spec'].values():
        if isinstance(v, str):
            if v not in s:
                s.add(v)
                print(v)
