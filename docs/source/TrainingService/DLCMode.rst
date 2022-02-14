**Run an Experiment on Aliyun PAI-DSW + PAI-DLC**
===================================================

NNI supports running an experiment on `PAI-DSW <https://help.aliyun.com/document_detail/194831.html>`__ , submit trials to `PAI-DLC <https://help.aliyun.com/document_detail/165137.html>`__ called dlc mode.

PAI-DSW server performs the role to submit a job while PAI-DLC is where the training job runs.

Setup environment
-----------------

Step 1. Install NNI, follow the install guide `here <../Tutorial/QuickStart.rst>`__.

Step 2. Create PAI-DSW server following this `link <https://help.aliyun.com/document_detail/163684.html?section-2cw-lsi-es9#title-ji9-re9-88x>`__. Note as the training service will be run on PAI-DLC, it won't cost many resources to run and you may just need a PAI-DSW server with CPU.

Step 3. Open PAI-DLC `here <https://pai-dlc.console.aliyun.com/#/guide>`__, select the same region as your PAI-DSW server. Move to ``dataset configuration`` and mount the same NAS disk as the PAI-DSW server does. (Note currently only PAI-DLC public-cluster is supported.)

Step 4. Open your PAI-DSW server command line, download and install PAI-DLC python SDK to submit DLC tasks, refer to `this link <https://help.aliyun.com/document_detail/203290.html>`__. Skip this step if SDK is already installed.


.. code-block:: bash

   wget https://sdk-portal-cluster-prod.oss-cn-zhangjiakou.aliyuncs.com/downloads/u-3536038a-3de7-4f2e-9379-0cb309d29355-python-pai-dlc.zip
   unzip u-3536038a-3de7-4f2e-9379-0cb309d29355-python-pai-dlc.zip
   pip install ./pai-dlc-20201203  # pai-dlc-20201203 refer to unzipped sdk file name, replace it accordingly.


Run an experiment
-----------------

Use ``examples/trials/mnist-pytorch`` as an example. The NNI config YAML file's content is like:

.. code-block:: yaml

  # working directory on DSW, please provie FULL path
  experimentWorkingDirectory: /home/admin/workspace/{your_working_dir}
  searchSpaceFile: search_space.json
  # the command on trial runner(or, DLC container), be aware of data_dir
  trialCommand: python mnist.py --data_dir /root/data/{your_data_dir}
  trialConcurrency: 1  # NOTE: please provide number <= 3 due to DLC system limit.
  maxTrialNumber: 10
  tuner:
    name: TPE
    classArgs:
      optimize_mode: maximize
  # ref: https://help.aliyun.com/document_detail/203290.html?spm=a2c4g.11186623.6.727.6f9b5db6bzJh4x
  trainingService:
    platform: dlc
    type: Worker
    image: registry-vpc.cn-beijing.aliyuncs.com/pai-dlc/pytorch-training:1.6.0-gpu-py37-cu101-ubuntu18.04
    jobType: PyTorchJob                             # choices: [TFJob, PyTorchJob]
    podCount: 1
    ecsSpec: ecs.c6.large
    region: cn-hangzhou
    nasDataSourceId: ${your_nas_data_source_id}
    accessKeyId: ${your_ak_id}
    accessKeySecret: ${your_ak_key}
    nasDataSourceId: ${your_nas_data_source_id}     # NAS datasource ID，e.g., datat56by9n1xt0a
    localStorageMountPoint: /home/admin/workspace/  # default NAS path on DSW
    containerStorageMountPoint: /root/data/         # default NAS path on DLC container, change it according your setting

Note: You should set ``platform: dlc`` in NNI config YAML file if you want to start experiment in dlc mode.

Compared with `LocalMode <LocalMode.rst>`__ training service configuration in dlc mode have these additional keys like ``type/image/jobType/podCount/ecsSpec/region/nasDataSourceId/accessKeyId/accessKeySecret``, for detailed explanation ref to this `link <https://help.aliyun.com/document_detail/203111.html#h2-url-3>`__.

Also, as dlc mode requires DSW/DLC to mount the same NAS disk to share information, there are two extra keys related to this: ``localStorageMountPoint`` and ``containerStorageMountPoint``.

Run the following commands to start the example experiment:

.. code-block:: bash

   git clone -b ${NNI_VERSION} https://github.com/microsoft/nni
   cd nni/examples/trials/mnist-pytorch

   # modify config_dlc.yml ...

   nnictl create --config config_dlc.yml

Replace ``${NNI_VERSION}`` with a released version name or branch name, e.g., ``v2.3``.

Monitor your job
----------------

To monitor your job on DLC, you need to visit `DLC  <https://pai-dlc.console.aliyun.com/#/jobs>`__ to check job status.
