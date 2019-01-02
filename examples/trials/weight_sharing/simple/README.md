# Weight Sharing in NNI
Many of the NAS(Neural Architecture Search) algorithms leverages the technique of weight sharing. For example, [DARTS][1] treats both model weights and network connection as trainable parameters; In [Morphism][2] algorithm, each new layer is inserted among existing trained layers and initialized in a way which is equivalent as before. Since trial is the basic unit of NNI's searching, we developed a way of sharing model weights across trials. 

## Dependencies
Currently we support weight sharing through NFS.
### NFS Setup
In NFS, files are physically stored on a server machine, and trials on the client machine can read/write those files in the same way that they access local files.
#### Install NFS on server machine
First, install NFS server:
```bash
sudo apt-get install nfs-kernel-server
```
Suppose `/tmp/nni/shared` is used as the physical storage, then run:
```bash
sudo mkdir -p /tmp/nni/shared
sudo echo "/tmp/nni/shared *(rw,sync,no_subtree_check,no_root_squash)" >> /etc/exports
sudo service nfs-kernel-server restart
```
You can check if the above directory is successfully exported by NFS using `sudo showmount -e localhost`

#### Install NFS on client machine
First, install NFS client:
```bash
sudo apt-get install nfs-common
```
Then create & mount the mounted directory of shared files:
```bash
sudo mkdir -p /mnt/nfs/nni/
sudo mount -t nfs 10.10.10.10:/tmp/nni/shared /mnt/nfs/nni
```
where `10.10.10.10` should be replaced by the real IP of NFS server machine in practice.
## Weight Sharing Example
Here we give an example of how to share files between different trials & machines, with [config file](./config.yml), [trial code](./main.py) and [tuner code](../../../tuners/weight_sharing/simple/simple_tuner.py). This example launches totally 4 trials, with 1 father trial and 3 child trials. The father generates a random file, then launch child trials to read & compute checksum of the file. Here child trials should wait until the father trial is done. So multiple thread mode should be enabled with `multithread: True` in `config.yml`, where tuner can schedule the trials using thread synchronization operations in python. 

[1]: https://arxiv.org/abs/1806.09055
[2]: https://arxiv.org/abs/1806.10282