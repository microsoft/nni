
from sdk.graph import Graph
from sdk.jit_optimzer.simple_strategy import SimpleOptimizationStrategy, dump_multi_graph
import json
from sdk.graph import _debug_dump_graph
import sdk

import subprocess
import os
import time
import argparse

PYTHON_PATH = '/home/zhenhua/py37/bin/python'


def _dump_dedup_trials(optimized_trials, trainer, batch_size=224):
    graphs_to_launch = {}
    for idx, graph_pack in enumerate(optimized_trials):
        code_path = f'artifact_live/micro_dedup/out{idx}.py'
        module_path = f'artifact_live.micro_dedup.out{idx}'
        graph_pack[0].generate_code(framework='pytorch', output_file=code_path)
        phy_graph = dump_multi_graph(graph_pack[0], graph_pack[1])
        graphs_to_launch[phy_graph['name']] = {
            'graph': phy_graph, 'code': code_path, 'module_path': module_path, 
            'trainer': trainer}
        graphs_to_launch[phy_graph['name']]['cmd_args'] = [
            f'--profile_out=artifact_live/micro_dedup/dedup_{idx}.perf', 
            f'--batch-size={batch_size}']
        filename = f'artifact_live/micro_dedup/out{idx}.json'
        with open(filename, 'w') as dump_file:
            json.dump(phy_graph, dump_file, indent=4)

    return graphs_to_launch


def _launch(graphs_to_launch, cpu_list='0-20', disable_distributed=False):
    processes = []
    for idx, graph_name in enumerate(graphs_to_launch):
        graph = graphs_to_launch[graph_name]['graph']
        if len(graph['graph']['inputs'][0]) == 0:
            graphs_to_launch[graph_name]['cmd_args'].append('--use_fake_input')
        module_path = graphs_to_launch[graph_name]['module_path']
        env = {}
        if 'env' in graph['configs']:
            env = graph['configs']['env'].copy()
            job_id = env['rank']
            if not disable_distributed:
                if 'is_distributed' not in env:
                    env['is_distributed'] = '1'
            else:
                if 'is_distributed' in env:
                    del env['is_distributed']

            for key in env:
                if not isinstance(env[key], str):
                    env[key] = str(env[key])
        else:
            job_id = 0
            env['CUDA_VISIBLE_DEVICES'] = str(idx % 4)
        if disable_distributed:
            graphs_to_launch[graph_name]['cmd_args'].append("--disable_shared")
        cmd = ['taskset', '-c', cpu_list, PYTHON_PATH, 
                graphs_to_launch[graph_name]['trainer'],
                f"--module_path={module_path}", f"--job_id={job_id}", 
                *graphs_to_launch[graph_name]['cmd_args']]
        #if eval:
        #    cmd.append('--eval')
        #    cmd.append('--resume')
        #if ckpt_dir:
        #    cmd.append(f'--ckpt_dir={ckpt_dir[idx]}')
        print(cmd, env)
        p = subprocess.Popen(cmd, env=env)
        processes.append(p)
    for p in processes:
        p.wait()
        rc = p.returncode
        assert(rc == 0)

#====================== Micro: Mnasnet (Figure 12) =============================
def micro_dedup_mnasnet(n_job):
    graphs = []
    for i in range(n_job):
        with open('./examples/graphs/mnasnet05-v2.json') as fp:
            g = Graph.load(json.load(fp))
            g.name = f'mnasnet_micro_{i}'
            graphs.append(g)
    if n_job <= 4:
        max_trial_per_gpu = 1
        max_mem_util = 1.01
    else:
        max_trial_per_gpu = 2
        max_mem_util = 2.01  # Hack to pack two jobs into one GPU
    sos = SimpleOptimizationStrategy({'use_multi_proc': True, 
                                        'max_mem_util': max_mem_util,
                                        'disable_merge': False, 
                                        'max_num_gpu': 4, 
                                        'max_trial_per_gpu': max_trial_per_gpu}, 
                                        disable_batch=True,
                                        disable_dedup=False)
    optimized_trials = sos.optimize(graphs)
    assert(len(optimized_trials) == 1)
    graphs_to_launch = _dump_dedup_trials(
        optimized_trials[0], "train_imagenet.py", batch_size=224)
    for g in graphs_to_launch:
        #graphs_to_launch[g]['cmd_args'].append('--use_dali')
        graphs_to_launch[g]['cmd_args'].append('--num-threads=8')
    _launch(graphs_to_launch)
    with open('artifact_live/micro_dedup/dedup_0.perf', 'r') as fp:
        tokens = fp.readlines()[0].split(' ')
        avg_speed = float(tokens[1])/float(tokens[0])
    # 224 is the batch size for mnasnet_0.5
    print(f"Throughput: {avg_speed*n_job*224} samples/s")


#====================== Micro: TextNAS (Figure 13) =============================
def micro_dedup_bert(n_job):
    graphs = []
    for i in range(n_job):
        with open('./examples/textnas/textnas_fix.json') as fp:
            g = Graph.load(json.load(fp))
            g.name = f'textnas_micro_{i}'
            graphs.append(g)
    if n_job <= 3:
        max_trial_per_gpu = 1
        max_mem_util = 1.01
    elif n_job <= 6:
        max_trial_per_gpu = 2
        max_mem_util = 2.01  # Hack to pack two jobs into one GPU
    elif n_job <= 9:
        max_trial_per_gpu = 3
        max_mem_util = 3.01
    elif n_job <= 12:
        max_trial_per_gpu = 4
        max_mem_util = 4.01

    # add  disable_dedup=True, disbale_standalone=True to run as baseline
    sos = SimpleOptimizationStrategy({'use_multi_proc': True, 
                                        'max_mem_util': max_mem_util, 
                                        'disable_merge': False,
                                        'max_num_gpu': 4, 
                                        'max_trial_per_gpu': max_trial_per_gpu}, 
                                        #disable_dedup=True,
                                        #disbale_standalone=True
                                        disable_batch=True
                                        )

    optimized_trials = sos.optimize(graphs)
    assert(len(optimized_trials) == 1)

    for trial in optimized_trials[0]:
        trial[0].configs['imports'] = [
            "op_libs.textnas_ops", "op_libs.textnas_utils"]

    graphs_to_launch = _dump_dedup_trials(
        optimized_trials[0], "train_sst.py", batch_size=128)
    for g in graphs_to_launch:
        if 'env' in graphs_to_launch[g]['graph']['configs']:
            job_id = int(graphs_to_launch[g]
                         ['graph']['configs']['env']['rank'])
            if job_id == 0:
                graphs_to_launch[g]['cmd_args'].append('--not_training')
    _launch(graphs_to_launch)
    with open('artifact_live/micro_dedup/dedup_0.perf', 'r') as fp:
        tokens = fp.readlines()[0].split(' ')
        avg_speed = float(tokens[1])/float(tokens[0])
    # 224 is the batch size for mnasnet_0.5
    print(f"Throughput: {avg_speed*n_job*128} samples/s")

#====================== Op Batching (Figure 14) ================================
def _gen_adapter(n_job):
    graphs = []
    for i in range(n_job):
        # with open('./examples/graphs/mnasnet05.json') as fp:
        with open(f'./generated/adapter.json') as fp:
            g = Graph.load(json.load(fp))
            for node in g.hidden_nodes:
                if 'parallel' not in node.name: 
                    node.set_attribute('non-trainable', True)
                else:
                    node.set_attribute('non-trainable', False)

            g.name = f'adapter-{i}'
            graphs.append(g)

    sos = SimpleOptimizationStrategy({'use_multi_proc': False, 
                                            'max_mem_util': 200.01, 
                                            'disable_merge': False,
                                            'max_num_gpu': 1, 
                                            'max_trial_per_gpu': 1000}, batch_size=8)
    optimized_trials = sos.optimize(graphs)
    return optimized_trials

def _profile_op_batch():
    
    import generated.op_batch as out
    import torch

    batch_size=8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = out.Graph()
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.125, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    for batch_idx in range(500):
        if batch_idx == 100:
            t_start = time.time()
        inputs = torch.FloatTensor(batch_size, 3, 32, 32).random_(0, 255)
        targets = torch.LongTensor(batch_size).random_(0, 10)
        #targets = torch.cat([targets]*num_batch)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        inputs, targets, outputs = net(inputs, targets)
        #print('size: ', outputs.size())
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(batch_idx)
    t_end = time.time()
    return (t_end-t_start)/400.0
        

def micro_op_batching(n_job):
    optimized_trials = _gen_adapter(n_job)
    assert(len(optimized_trials) == 1)

    for trial_pack in optimized_trials[0]:
        with open('dump_batch.json', 'w') as fp:
            json.dump(trial_pack[0].dump(), fp)
        trial_pack[0].generate_code(framework='pytorch', output_file='./generated/op_batch.py')
    print("Start Job")
    iter_time = _profile_op_batch()
    total_throughput = 1.0/iter_time*n_job*8
    print(f"Throughput: {total_throughput} samples/s")
    

#====================== Mnasnet E2E (Figure 16a) ===============================
def test_mnasnet_list(job_list, perf_dir='v100perf', plan_path='mnaset.plan'):
    if not os.path.exists(perf_dir):
        os.makedirs(perf_dir)
    graphs = []
    for i in job_list:
        # with open('./examples/graphs/mnasnet05.json') as fp:
        with open(f'./generated/mnasnet-{i}.json') as fp:
            g = Graph.load(json.load(fp))
            g.name = f'mnasnet-{i}'
            graphs.append(g)
    n_job = len(job_list)
    if n_job <= 4:
        max_trial_per_gpu = 1
        max_mem_util = 1.01
    else:
        max_trial_per_gpu = 10
        max_mem_util = 0.9 # Hack to pack two jobs into one GPU
    sos = SimpleOptimizationStrategy({'use_multi_proc': True, 
                                        'max_mem_util': max_mem_util, 
                                        'disable_merge': False,
                                        'max_num_gpu': 4, 
                                        'max_trial_per_gpu': max_trial_per_gpu},
                                        disable_dedup=False,
                                        disable_batch=True)
    optimized_trials = sos.optimize(graphs)
    #assert(len(optimized_trials) == 1)
    for launch_id, grouped_trials in enumerate(optimized_trials):
        grouped_graph_names = [
            (_[0].name, _[0].configs['env']['DEVICE_ID']) 
                for _ in grouped_trials
        ]
        n_graphs = len(grouped_graph_names)
        with open(plan_path, 'a') as fp:
            fp.write(",".join([f'{_[0]}|{_[1]}' for _ in grouped_graph_names]))
            fp.write('\n')
        graphs_to_launch = _dump_dedup_trials(
            grouped_trials, "train_imagenet.py", batch_size=32)
        try:
            _launch(graphs_to_launch, disable_distributed=False)
        except:
            raise ValueError(f'None-zero exit code in {job_list}')
        with open('artifact_live/micro_dedup/dedup_0.perf', 'r') as fp:
            tokens = fp.readlines()[0].split(' ')
            avg_speed = float(tokens[1])/float(tokens[0])
        for idx, name in enumerate(grouped_graph_names):
            os.system(
                f"cp artifact_live/micro_dedup/dedup_{idx}.perf {perf_dir}/{name[0]}")
        # 224 is the batch size for mnasnet_0.5
        print(f"Throughput: {avg_speed*n_graphs*32} samples/s")



#====================== SPOS (Figure 17)  ======================================

def _gen_spos_super_graph(n_job):
    from op_libs.spos import ShuffleNetV2OneShot, ModelTrain
    from sdk.mutators.builtin_mutators import ModuleMutator

    base_model = ShuffleNetV2OneShot()
    exp = sdk.create_experiment('spos', base_model)
    exp.specify_training(ModelTrain)

    mutators = []
    for i in range(20):
        mutator = ModuleMutator('features.'+str(i), [{'ksize': 3}, 
                                {'ksize': 5}, {'ksize': 7}, 
                                {'ksize': 3, 
                                'sequence': 'dpdpdp'}])
        mutators.append(mutator)
    exp.specify_mutators(mutators)
    exp.specify_strategy('naive.strategy.main', 'naive.strategy.RandomSampler')
    run_config = {
        'authorName': 'nas',
        'experimentName': 'nas',
        'trialConcurrency': 1,
        'maxExecDuration': '24h',
        'maxTrialNum': 999,
        'trainingServicePlatform': 'local',
        'searchSpacePath': 'empty.json',
        'useAnnotation': False
    } # nni experiment config
    pre_run_config = {
        'name' : f'spos',
        'x_shape' : [256, 3 , 244, 244],
        'x_dtype' : 'torch.float32',
        'y_shape' : [256],
        "y_dtype" : "torch.int64",
        "mask" : False,
        "imports" : ["op_libs.spos"]
    }
    exp.run(run_config, pre_run_config=pre_run_config)

def e2e_spos(n_job):
    _gen_spos_super_graph(n_job)
    assert(n_job <= 4)  # TODO: for test in one server use 4 jobs at most
    graphs_to_launch = {}
    for i in range(n_job):
        graph_name = 'spos_'+str(i)
        graphs_to_launch[graph_name] = {}
        graphs_to_launch[graph_name]['graph'] = \
            {"name": graph_name, 
            "graph": {
                "inputs":[[None]],  # add an element to avoid fake_input
                "outputs":[[None]], 
                "hidden_nodes":[], 
                "edges":[]},
            "configs": {
                "env": {
                    'is_distributed': '1', 
                    'rank': str(i), 'distributed_backend': 'nccl', 
                    'world_size': str(n_job), 
                    'DEVICE_ID': str(i)
                    }
                }
            }
        graphs_to_launch[graph_name]['configs']={
            "imports" : ["op_libs.spos"]
            }
        graphs_to_launch[graph_name]['module_path'] = 'generated.spos'
        graphs_to_launch[graph_name]['trainer'] = 'train_imagenet.py'
        graphs_to_launch[graph_name]['cmd_args']= \
            ['--mix_para',\
                 f'--batch-size={256}', '--use_dali', f'--learning-rate={0.125*n_job}']

    _launch(graphs_to_launch)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('expt_name', metavar='EXPT',
                    help='experiment name: [micro_dedup_cpu, \
                        micro_dedup_gpu, micro_batching, \
                        e2e_dali_mnasnset, e2e_spos_mix_parallel]')
parser.add_argument('--n', default=8, type=int,
                    help='number of jobs to run')

if __name__ == '__main__':
    args = parser.parse_args()
    if args.expt_name == 'micro_dedup_cpu':
        micro_dedup_mnasnet(args.n) # n=8 for artifact
    elif args.expt_name == 'micro_dedup_gpu':
        micro_dedup_bert(args.n) # n=12 for artifact
    elif args.expt_name == 'micro_batching':
        micro_op_batching(args.n) # n=192 for artifact
    elif args.expt_name == 'e2e_dali_mnasnet':
        mnasnet_plan_path = 'merged_perf/mnasnet.plan'
        with open(mnasnet_plan_path, 'w') as fp:
            fp.write('mnanset\n')
        for i in range(20): # 20 batches, each batch has 50 graphs
            job_list = [_ for _ in range(i*50, i*50+50)]
            test_mnasnet_list(job_list, 
                            perf_dir="merged_perf",
                            plan_path=mnasnet_plan_path)
        P = subprocess.Popen([PYTHON_PATH, 'fast_scheduler.py'])
    elif args.expt_name == 'e2e_spos_mix_parallel':
        e2e_spos(args.n) # n=4 for artifact
    # elif args.expt_name == 'spos_eval':
    #     e2e_spos(4, eval=False) # n=4 for artifact
