# Data Management in NNI
## overview
In NNI's experiments, users should specify a `codeDir` configuration, this configuration contains the code files and data files, NNI will use data in this folder to start trial jobs. Generally, trial jobs will share same `codeDir` folder in NNI, and do not write their own data to this folder to keep original data is not changed. 

## environment variable
NNI has environment variables including `$NNI_CODE_DIR`, `$NNI_SYS_DIR` and `$NNI_OUTPUT_DIR` to specify different data path. Users could get these environment variables from their python code, for example:
```
import os
print(os.environ['NNI_OUTPUT_DIR'])
```

1. `$NNI_CODE_DIR`.  
    `$NNI_CODE_DIR` is a path which stores users' code data, trial jobs share same code in this path.
2. `$NNI_SYS_DIR`.  
    `$NNI_SYS_DIR`  is a path which stores trial jobs' metirc file and parameter file.
3. `$NNI_OUTPUT_DIR`.  
    `NNI_OUTPUT_DIR`  is a path which stores trial jobs' output file, including `stdout`, `stderr` etc.

![](../../img/nni_data_management.jpg)


## local platform
In local platform, NNI will use `codeDir` folder which user specified directly.Every trial job will share same `codeDir` folder, and does not write output data to this folder. 
## remote platform
In remote platform, NNI will upload `codeDir` to different remtoe machines, the trial jobs in these machine could share the `codeDir` folder.
## PAI, Kubeflow, FrameworkController
In these platforms NNI will upload data in `codeDir` to storage, and different trial jobs will mount this folder to their container. 