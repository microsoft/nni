# 高级神经网络架构搜索教程

目前，许多 NAS（Neural Architecture Search，神经网络架构搜索）算法都在尝试上使用了 **权重共享（weight sharing）** 的方法来加速训练过程。 例如，[ENAS](https://arxiv.org/abs/1802.03268) 与以前的 [NASNet](https://arxiv.org/abs/1707.07012) 算法相比，通过'*子模型间的参数共享（parameter sharing between child models）*'提高了 1000 倍的效率。 而例如 [DARTS](https://arxiv.org/abs/1806.09055), [Network Morphism](https://arxiv.org/abs/1806.10282), 和 [Evolution](https://arxiv.org/abs/1703.01041) 等算法也利用或者隐式的利用了权重共享。

这是关于如何在 NNI 中启用权重共享的教程。

## 尝试间的权重共享

目前，推荐通过 NFS （Network File System）来进行权重共享，它是轻量、相对高效的多机共享文件方案。 欢迎社区来共享更多高效的技术。

### 通过 NFS 文件的权重共享

使用 NFS 配置（见下文），尝试代码可以通过读写文件来共享模型权重。 建议使用调参器的存储路径：

```yaml
tuner:
  codeDir: path/to/customer_tuner
  classFileName: customer_tuner.py 
  className: CustomerTuner
  classArgs:
    ...
    save_dir_root: /nfs/storage/path/
```

并让调参器来决定在什么路径读写权重文件，通过 `nni.get_next_parameters()` 来获取路径：

![weight_sharing_design](./img/weight_sharing.png)

例如，在 Tensorflow 中：

```python
# 保存 models
saver = tf.train.Saver()
saver.save(sess, os.path.join(params['save_path'], 'model.ckpt'))
# 读取 models
tf.init_from_checkpoint(params['restore_path'])
```

超参中的 `'save_path'` 和 `'restore_path'` 可以通过调参器来管理。

### NFS 配置

NFS follows the Client-Server Architecture, with an NFS server providing physical storage, trials on the remote machine with an NFS client can read/write those files in the same way that they access local files.

#### NFS Server

An NFS server can be any machine as long as it can provide enough physical storage, and network connection with **remote machine** for NNI trials. Usually you can choose one of the remote machine as NFS Server.

On Ubuntu, install NFS server through `apt-get`:

```bash
sudo apt-get install nfs-kernel-server
```

Suppose `/tmp/nni/shared` is used as the physical storage, then run:

```bash
mkdir -p /tmp/nni/shared
sudo echo "/tmp/nni/shared *(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
sudo service nfs-kernel-server restart
```

You can check if the above directory is successfully exported by NFS using `sudo showmount -e localhost`

#### NFS Client

For a trial on remote machine able to access shared files with NFS, an NFS client needs to be installed. For example, on Ubuntu:

```bash
sudo apt-get install nfs-common
```

Then create & mount the mounted directory of shared files:

```bash
mkdir -p /mnt/nfs/nni/
sudo mount -t nfs 10.10.10.10:/tmp/nni/shared /mnt/nfs/nni
```

where `10.10.10.10` should be replaced by the real IP of NFS server machine in practice.

## 尝试依赖控制的异步调度模式

多机时启用权重的尝试，大部分情况是通过保证**先写后读**的方式来保持一致性。 子节点在父节点的尝试完成训练前，不应该读取父节点模型。 要解决这个问题，要通过 `multiThread: true` 来启用**异步调度模式**。在 `config.yml` 中，每次收到 `NEW_TRIAL` 请求，分派一个新的调参器线程时，调参器线程可以决定是否阻塞当前线程。 For example:

```python
    def generate_parameters(self, parameter_id):
        self.thread_lock.acquire()
        indiv = # 新尝试的配置
        self.events[parameter_id] = threading.Event()
        self.thread_lock.release()
        if indiv.parent_id is not None:
            self.events[indiv.parent_id].wait()

    def receive_trial_result(self, parameter_id, parameters, reward):
        self.thread_lock.acquire()
        # 处理尝试结果的配置
        self.thread_lock.release()
        self.events[parameter_id].set()
```

## 样例

详细用法，请参考 [简单权重共享样例](../test/async_sharing_test)。 还有根据 [ga_squad](../examples/trials/ga_squad) 改动的阅读理解的[实际样例](../examples/trials/weight_sharing/ga_squad)。