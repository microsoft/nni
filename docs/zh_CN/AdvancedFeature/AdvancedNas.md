# 高级神经网络架构搜索教程

目前，许多 NAS（Neural Architecture Search，神经网络架构搜索）算法都在 Trial 上使用了 **权重共享（weight sharing）** 的方法来加速训练过程。 例如，[ENAS](https://arxiv.org/abs/1802.03268) 与以前的 [NASNet](https://arxiv.org/abs/1707.07012) 算法相比，通过'*子模型间的参数共享（parameter sharing between child models）*'提高了 1000 倍的效率。 而例如 [DARTS](https://arxiv.org/abs/1806.09055), [Network Morphism](https://arxiv.org/abs/1806.10282), 和 [Evolution](https://arxiv.org/abs/1703.01041) 等算法也利用或者隐式的利用了权重共享。

本教程介绍了如何使用权重共享。

## 权重共享

推荐通过 NFS （Network File System）进行权重共享，它是轻量、相对高效的多机共享文件方案。 欢迎社区来共享更多高效的技术。

### 通过 NFS 文件使用权重共享

使用 NFS 配置（见下文），Trial 代码可以通过读写文件来共享模型权重。 建议使用 Tuner 的存储路径：

```yaml
tuner:
  codeDir: path/to/customer_tuner
  classFileName: customer_tuner.py
  className: CustomerTuner
  classArgs:
    ...
    save_dir_root: /nfs/storage/path/
```

并让 Tuner 来决定在什么路径读写权重文件，通过 `nni.get_next_parameters()` 来获取路径：

<img src="https://user-images.githubusercontent.com/23273522/51817667-93ebf080-2306-11e9-8395-b18b322062bc.png" alt="drawing" width="700" />

例如，在 Tensorflow 中：

```python
# 保存 models
saver = tf.train.Saver()
saver.save(sess, os.path.join(params['save_path'], 'model.ckpt'))
# 读取 models
tf.init_from_checkpoint(params['restore_path'])
```

超参中的 `'save_path'` 和 `'restore_path'` 可以通过 Tuner 来管理。

### NFS 配置

NFS 使用了客户端/服务器架构。通过一个 NFS 服务器来提供物理存储，远程计算机上的 Trial 使用 NFS 客户端来读写文件，操作上和本地文件相同。

#### NFS 服务器

如果有足够的存储空间，并能够让 NNI 的 Trial 通过**远程机器**来连接，NFS 服务可以安装在任何计算机上。 通常，可以选择一台远程服务器作为 NFS 服务。

在 Ubuntu 上，可通过 `apt-get` 安装 NFS 服务：

```bash
sudo apt-get install nfs-kernel-server
```

假设 `/tmp/nni/shared` 是物理存储位置，然后运行：

```bash
mkdir -p /tmp/nni/shared
sudo echo "/tmp/nni/shared *(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
sudo service nfs-kernel-server restart
```

可以通过命令 `sudo showmount -e localhost` 来检查上述目录是否通过 NFS 成功导出了

#### NFS 客户端

为了通过 NFS 访问远程共享文件，需要安装 NFS 客户端。 例如，在 Ubuntu 上运行：

```bash
sudo apt-get install nfs-common
```

然后创建并装载上共享目录：

```bash
mkdir -p /mnt/nfs/nni/
sudo mount -t nfs 10.10.10.10:/tmp/nni/shared /mnt/nfs/nni
```

实际使用时，IP `10.10.10.10` 需要替换为 NFS 服务器的真实地址。

## Trial 依赖控制的异步调度模式

多机间启用权重的 Trial，一般是通过**先写后读**的方式来保持一致性。 子节点在父节点的 Trial 完成训练前，不应该读取父节点模型。 要解决这个问题，要通过 `multiThread: true` 来启用**异步调度模式**。在 `config.yml` 中，每次收到 `NEW_TRIAL` 请求，分派一个新的 Trial 时，Tuner 线程可以决定是否阻塞当前线程。 例如：

```python
    def generate_parameters(self, parameter_id):
        self.thread_lock.acquire()
        indiv = # 新 Trial 的配置
        self.events[parameter_id] = threading.Event()
        self.thread_lock.release()
        if indiv.parent_id is not None:
            self.events[indiv.parent_id].wait()

    def receive_trial_result(self, parameter_id, parameters, reward):
        self.thread_lock.acquire()
        # 处理 Trial 结果的配置
        self.thread_lock.release()
        self.events[parameter_id].set()
```

## 样例

详细内容参考：[简单的参数共享样例](https://github.com/Microsoft/nni/tree/master/test/async_sharing_test)。 基于已有的 [ga_squad](https://github.com/Microsoft/nni/tree/master/examples/trials/ga_squad) 样例，还提供了新的 [样例](https://github.com/Microsoft/nni/tree/master/examples/trials/weight_sharing/ga_squad)。