from queue import PriorityQueue

import argparse
import os
import math
# from matplotlib import pyplot as plt 

class Job:
  def __init__(self, name, confs, num_gpu, batch_time, gpu_map=None):
    self.name = name
    self.confs = confs.copy()
    self.batch_time = batch_time
    self.speed = 1.0/batch_time
    self.num_gpu = num_gpu
    self.gpu_map = gpu_map
  
  def __lt__(self, other):
    return self.name < other.name
    
class Scheduler:
  def __init__(self, all_jobs, num_worker, num_gpu_per_worker, num_batch_conf, batch_size, num_sample):
    self.all_jobs = all_jobs
    self.num_worker = num_worker
    self.num_gpu_per_worker = num_gpu_per_worker
    self.num_batch_conf = num_batch_conf
    self.batch_size = batch_size
    self.num_sample = num_sample
  
    self.worker_status = {i:{g:None for g in range(num_gpu_per_worker)} for i in range(num_worker)}
    self.job_idx = 0
    self.finished_jobs = []
    self.running_jobs = set()
    self.current_batch = []
    self.event_queue = PriorityQueue()
    self.placement = {}
    self.current_time = 0

    self.plot_x = []
    self.plot_y = []
  
  def _count_free_gpu(self, worker_id):
    ret = 0
    for g in range(self.num_gpu_per_worker):
      if self.worker_status[worker_id][g] == None:
        ret += 1
    return ret

  def _schedule_job(self, job):
    gpus = []
    for worker_id in range(self.num_worker):
      if self._count_free_gpu(worker_id) >= job.num_gpu:
        gpus = [(worker_id, gpu_id) for gpu_id in range(self.num_gpu_per_worker) if self.worker_status[worker_id][gpu_id] == None]
        gpus = gpus[0:job.num_gpu]
    if len(gpus) >= job.num_gpu:
      assert len(gpus) == job.num_gpu
      self.placement[job.name] = gpus.copy()
      for worker_id, gpu_id in gpus:
        self.worker_status[worker_id][gpu_id] = job
      print("[%.4f] Job %s is Scheduled on %s" % (self.current_time, job.name, str(self.placement[job.name])))
      return True
    else:
      return False

  def _try_schedule(self):
    while True:
      succ = False
      for j in self.current_batch:
        succ = self._schedule_job(j)
        if succ:
          self.event_queue.put(\
            (self.current_time+j.batch_time*math.ceil(float(self.num_sample)/self.batch_size), 
            j)
          )
          self.running_jobs.add(j)
          self.current_batch.remove(j)
          break
      if not succ:
        break 
  
  def plot_mark(self, offset= 0):
    self.plot_x.append(self.current_time + offset)
    self.plot_y.append(sum([self.num_gpu_per_worker- self._count_free_gpu(worker_id) for worker_id in range(self.num_worker)]))

  def run(self):
    while len(self.finished_jobs) < len(self.all_jobs):
      if len(self.current_batch) == 0 and len(self.running_jobs) == 0:
        cnt_conf = 0
        while self.job_idx < len(self.all_jobs) and cnt_conf < self.num_batch_conf:
          cnt_conf += len(self.all_jobs[self.job_idx].confs)
          self.current_batch.append(self.all_jobs[self.job_idx])
          self.job_idx += 1
      if self.event_queue.qsize() == 0:
        self.plot_mark(offset = 0.001)
        self._try_schedule()
        self.plot_mark()
      else:
        e = self.event_queue.get()
        self.current_time = e[0]
        j = e[1]
        self.finished_jobs.append(j)
        self.running_jobs.remove(j)
        self.plot_mark(offset = 0.001)
        for worker_id, gpu_id in self.placement[j.name]:
          assert self.worker_status[worker_id][gpu_id] == j
          self.worker_status[worker_id][gpu_id] = None
        
        self._try_schedule()
        self.plot_mark()
    return self.current_time

def extract_job_from_plan(logdir, model_space, plan, metrics):
  all_jobs = []
  with open(plan, 'r') as fp_plan:
    lines = fp_plan.readlines()
    for idx, line in enumerate(lines[1:]):
      #tokens = line.strip().split(':')
      #if len(tokens) == 0:
      #  break
      #job_tokens = tokens[0].split('-')
      job_name = f'J{idx}'#job_tokens[0]
      #num_gpu = int(job_tokens[1])
      tokens = line.split(',')
      confs = []
      batch_time = 0
      gpu_map = {}
      used_gpus = set()
      for m in tokens:
        if len(m) > 0:
          model_tokens = m.split('|')
          conf_id = model_tokens[0]
          gpu_id = int(model_tokens[1])
          confs.append(conf_id)
          gpu_map[conf_id] = gpu_id
          used_gpus.add(gpu_id)
      batch_time = max([metrics[conf_id]['avg_batch_time'] for conf_id in confs])
      job = Job(job_name, confs, len(used_gpus), batch_time, gpu_map=gpu_map)
      all_jobs.append(job)
  return all_jobs

def get_metrics(path):
  with open(path, 'r') as fp:
    lines = fp.readlines()
    tokens = lines[0].strip().split(' ')
    metrics = {'avg_batch_time': float(tokens[0])/float(tokens[1])}
  return metrics

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--log-dir", type=str, default="merged_perf")
  parser.add_argument("--num_conf", type=int, default=1000)
  parser.add_argument("--model_space", type=str, default='mnasnet')
  parser.add_argument("--num_gpu_per_worker", type=int, default=4)
  parser.add_argument("--num_worker", type=int, default=1)
  parser.add_argument("--num_batch_conf", type=int, default=50)
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--num_sample", type=int, default=1281167)
  parser.add_argument("--plan", type=str, default='merged_perf/mnasnet.plan')

  args = parser.parse_args()
  mode = 'single' if args.plan == 'None' else 'multi'
  
  metrics = {}
  for conf_id in range(args.num_conf):
    path = os.path.join(args.log_dir, "%s-%d" % (args.model_space, conf_id))
    metrics[f'{args.model_space}-{conf_id}'] = get_metrics(path)
  
  all_jobs = []
  if mode == 'single':
    for conf_id in range(args.num_conf):
      j = Job('J%d' % conf_id, [conf_id], 1, metrics[conf_id]['avg_batch_time'])
      all_jobs.append(j)
  elif mode == 'multi':
    all_jobs = extract_job_from_plan(args.log_dir, args.model_space, args.plan, metrics)
  
  schd = Scheduler(all_jobs, args.num_worker, args.num_gpu_per_worker, args.num_batch_conf, args.batch_size, args.num_sample)
  makespan = schd.run()
  print(f'{makespan/3600.0} hours for {args.model_space} w/ BS={args.batch_size}')

  # plt.plot(schd.plot_x, schd.plot_y)
  # plt.show()