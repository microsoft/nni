import os

profile_graphs_fp = "profiled_data"
class Profiler:
    def __init__(self):
        self.log_metrics = {}
        self.load()
    
    def load(self):
        with open(profile_graphs_fp+'/perf_list.txt', 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                tokens = line.rstrip().split(' ')
                graph_name = tokens[0]
                profile_path = tokens[1]
                self.log_metrics[graph_name] = self._get_metrics(os.path.join(profile_graphs_fp, profile_path))
    
    def profile(self, graph):
        if graph.name in self.log_metrics:
            return self.log_metrics[graph.name]
        else:
            #TODO: integrate job profiling into Retiarii
            return {"mem_util" : 0.9, "avg_batch_time": 1.0}
                
    def _get_metrics(self, path):
        metrics = {}
        with open(path+'-monitor', 'r') as fp:
            lines = fp.readlines()
            tokens = lines[0].split(' ')
            metrics['gpu_util'] = float(tokens[2])
            metrics['mem_util'] = float(tokens[3])

        with open(path+'-perf', 'r') as fp:
            lines = fp.readlines()
            tokens = lines[1].strip().split(',')
            metrics['data_time'] = float(tokens[0])
            metrics['batch_time'] = float(tokens[1])
            tokens = lines[0].strip().split(',')
            metrics['avg_batch_time'] = float(tokens[1])/float(tokens[0])
        return metrics