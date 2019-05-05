# Hyperparameter Optimization Comparison

Comparison of Hyperparameter Optimization algorithms on several problems.

Hyperparameter Optimization algorithms are list below:

- Random Search
- Grid Search
- Evolution
- Anneal
- Metis
- TPE
- SMAC
- HyperBand
- BOHB

All algorithms run in NNI local environment.

## First Fake Function

### Problem Description

Firstly, one fake function is made to test the effectiveness in convex problem when hyperparameters are not independent.

Fake Function:

![](<http://latex.codecogs.com/gif.latex?Loss(x,y,z)=(x-30+0.1)^2\times(y-40+0.1)^2\times(x+z-50+0.1)^2>)

### Search Space

```json
{
  "x": {
    "_type": "randint",
    "_value": [100]
  },
  "y": {
    "_type": "randint",
    "_value": [100]
  },
  "z": {
    "_type": "randint",
    "_value": [100]
  }
}
```

The total search space is 1,000,000, the number of maximum trial to 1000. The time limitation is 24 hours.

### Results

We repeat these algorithms (Random Search, Grid Search, Evolution, Anneal, Metis, TPE, SMAC) for three times and evaluate these using TOP 10 results.

| Algorithm                  | TOP1 Loss | **TOP2** Loss | **TOP3** Loss | **TOP4** Loss | **TOP5 Loss** | **TOP6 Loss** | **TOP7 Loss** | **TOP8** Loss | **TOP9 **Loss | **TOP10** Loss |
| -------------------------- | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------- |
| Random Search              | 39.4258   | 43.5732       | 43.5732       | 103.612       | 142.110       | 186.896       | 244.015       | 244.328       | 286.591       | 314.672        |
| Random Search              | 0.67076   | 1.79828       | 8.24264       | 8.34632       | 21.538        | 44.475        | 45.846        | 65.593        | 80.8021       | 101.787        |
| Random Search              | 1.79828   | 341.177       | 545.175       | 548.543       | 723.663       | 749.609       | 750.266       | 962.922       | 1209.23       | 1394.350       |
| Grid Search (All the same) | 3648.28   | 4005.75       | 4379.92       | 4770.80       | 5178.38       | 5602.67       | 6043.66       | 6501.35       | 6975.75       | 7466.86        |
| Evolution                  | **1e-06** | **1e-06**     | 0.000121      | 0.000361      | 0.000361      | 0.000441      | 0.000841      | 0.000841      | 0.001521      | 0.002601       |
| Evolution                  | 8.1e-05   | 8.1e-05       | 8.1e-05       | 8.1e-05       | 0.000361      | 0.000361      | 0.000441      | 0.000441      | 0.000841      | 0.000841       |
| Evolution                  | 0.000121  | 0.000841      | 0.001521      | 0.002401      | 0.002601      | 0.002601      | 0.003481      | 0.003481      | 0.003481      | 0.003481       |
| Anneal                     | 0.009801  | 0.009801      | 0.014641      | 0.029241      | 0.036481      | 0.123201      | 0.184041      | 0.346921      | 0.385641      | 0.421201       |
| Anneal                     | **1e-06** | 0.000841      | 0.000961      | 0.000961      | 0.006561      | 0.014641      | 0.014641      | 0.025281      | 0.025921      | 0.029241       |
| Anneal                     | 0.000361  | 0.000961      | 0.001521      | 0.004761      | 0.019321      | 0.251001      | 0.303601      | 0.326041      | 0.349281      | 0.385641       |
| Metis                      | 211.440   | 1822.03       | 2780.55       | 2978.86       | 17829.99      | 24463.77      | 39732.05      | 42362.28      | 44995.31      | 45105.68       |
| Metis                      | 0.241081  | 219.069       | 522.991       | 1331.59       | 1670.275      | 2733.302      | 2769.811      | 3698.950      | 5623.350      | 6162.83        |
| Metis                      | 54.479    | 749.171       | 1031.758      | 1349.019      | 1349.019      | 1380.048      | 2449.359      | 2919.63       | 3974.167      | 5581.733       |
| TPE                        | 0.000441  | 0.007921      | 0.014641      | 0.016641      | 0.040401      | 0.044521      | 0.058081      | 0.123201      | 0.152881      | 0.261121       |
| TPE                        | 0.006241  | 0.116281      | 0.210681      | 0.231361      | 1.846881      | 2.253001      | 2.499561      | 3.964081      | 5.764801      | 8.242641       |
| TPE                        | 0.029241  | 0.194481      | 0.281961      | 0.301401      | 0.385641      | 1.718721      | 2.927521      | 5.248681      | 5.948721      | 7.338681       |
| SMAC                       | **1e-6**  | 0.004761      | 0.008281      | 0.019321      | 0.019881      | 0.053361      | 0.212521      | 0.212521      | 0.212521      | 0.212521       |
| SMAC                       | 0.000841  | 0.004761      | 0.004761      | 0.009801      | 0.421201      | 0.505521      | 0.576081      | 0.707281      | 0.793881      | 2.099601       |
| SMAC                       | 0.000121  | 0.000121      | 0.000841      | 0.004761      | 0.014641      | 0.022801      | 0.029241      | 0.040401      | 0.090601      | 0.101761       |

For Metis, there are about 300 trials because it runs slowly due to its high time complexity O(n^3) in Gaussian Process.

## Second Fake Function

### Problem Description

Similar to first one, another fake function is made to test the effectiveness in convex problem when hyperparameters are independent.

Fake Function:

![](http://latex.codecogs.com/gif.latex?Loss(x,y,z)=(x-30+0.1)^2 +(y-40+0.1)^2+(z-50+0.1)^2)

### Search Space

```json
{
  "x": {
    "_type": "randint",
    "_value": [100]
  },
  "y": {
    "_type": "randint",
    "_value": [100]
  },
  "z": {
    "_type": "randint",
    "_value": [100]
  }
}
```

The total search space is 1,000,000, we set the number of maximum trial to 1000. The time limitation is 24 hours.

### Results

We repeat these algorithms (Random Search, Grid Search, Evolution, Anneal, Metis, TPE, SMAC) for three times and evaluate these using TOP 10 results.

| Algorithm                 | TOP1 Loss | **TOP2 Loss** | **TOP3 **Loss | **TOP4 **Loss | **TOP5** Loss | **TOP6 **Loss | **TOP7 **Loss | **TOP8 Loss** | **TOP9** Loss | **TOP10 **Loss |
| ------------------------- | --------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------- |
| Random Search             | 44.03     | 50.03         | 65.63         | 80.43         | 98.83         | 122.43        | 126.83        | 135.63        | 137.63        | 139.23         |
| Random Search             | 26.43     | 52.43         | 65.63         | 68.03         | 83.23         | 88.83         | 98.03         | 102.43        | 125.63        | 157.63         |
| Random Search             | 14.43     | 31.63         | 92.03         | 121.63        | 146.03        | 147.63        | 169.63        | 170.03        | 171.23        | 176.43         |
| Grid Search(All the same) | 1272.03   | 1272.83       | 1273.23       | 1275.62       | 1276.43       | 1280.43       | 1281.63       | 1287.23       | 1288.83       | 1296.03        |
| Evolution                 | 0.83      | 1.63          | 1.63          | 1.63          | 2.03          | 2.03          | 2.43          | 2.43          | 2.83          | 4.43           |
| Evolution                 | 3.23      | 4.43          | 4.43          | 5.23          | 6.03          | 6.03          | 10.03         | 10.83         | 11.23         | 11.63          |
| Evolution                 | 2.43      | 3.63          | 6.03          | 6.43          | 8.83          | 9.63          | 10.83         | 12.03         | 13.63         | 18.43          |
| Anneal                    | 17.63     | 20.03         | 26.03         | 27.23         | 28.43         | 33.23         | 34.03         | 38.83         | 42.43         | 53.23          |
| Anneal                    | 18.03     | 19.23         | 26.03         | 29.63         | 38.03         | 45.63         | 54.03         | 56.43         | 60.83         | 74.03          |
| Anneal                    | 4.43      | 14.03         | 18.83         | 19.23         | 68.03         | 76.43         | 84.03         | 90.03         | 92.43         | 107.63         |
| Metis                     | 2.03      | 2.83          | 5.23          | 18.03         | 21.63         | 22.43         | 26.83         | 28.43         | 36.03         | 37.23          |
| Metis                     | 17.63     | 21.23         | 27.23         | 29.23         | 29.63         | 33.23         | 34.83         | 36.43         | 36.83         | 38.03          |
| Metis                     | 14.03     | 18.03         | 23.23         | 27.23         | 27.63         | 28.03         | 28.03         | 28.03         | 47.63         | 52.83          |
| TPE                       | 0.83      | 16.83         | 18.03         | 18.43         | 27.63         | 33.23         | 33.63         | 47.63         | 48.83         | 52.03          |
| TPE                       | 3.23      | 6.43          | 6.43          | 9.63          | 10.03         | 13.23         | 13.63         | 14.83         | 17.23         | 21.23          |
| TPE                       | 13.23     | 26.03         | 28.83         | 36.83         | 51.63         | 58.83         | 66.83         | 83.63         | 86.43         | 87.23          |
| SMAC                      | **0.03**  | 1.23          | 1.23          | 1.23          | 1.23          | 2.03          | 2.43          | 2.83          | 3.23          | 5.63           |
| SMAC                      | **0.03**  | 1.23          | 1.23          | 1.23          | 1.23          | 2.03          | 2.43          | 2.83          | 3.23          | 5.63           |
| SMAC                      | **0.03**  | 1.23          | 1.23          | 1.23          | 1.23          | 2.03          | 2.43          | 2.83          | 3.23          | 5.63           |

For Metis, there are about 300 trials because it runs slowly due to its high time complexity O(n^3) in Gaussian Process.

## AutoGBDT Example

### Problem Description

Nonconvex problem on the hyper-parameter search of AutoGBDT example.

### Search Space

```json
{
  "num_leaves": {
    "_type": "choice",
    "_value": [10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 48, 64, 96, 128]
  },
  "learning_rate": {
    "_type": "choice",
    "_value": [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
  },
  "max_depth": {
    "_type": "choice",
    "_value": [
      -1,
      2,
      3,
      4,
      5,
      6,
      8,
      10,
      12,
      14,
      16,
      18,
      20,
      22,
      24,
      28,
      32,
      48,
      64,
      96,
      128
    ]
  },
  "feature_fraction": {
    "_type": "choice",
    "_value": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
  },
  "bagging_fraction": {
    "_type": "choice",
    "_value": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
  },
  "bagging_freq": {
    "_type": "choice",
    "_value": [1, 2, 4, 8, 10, 12, 14, 16]
  }
}
```

The total search space is 1,204,224, we set the number of maximum trial to 1000. The time limitation is 48 hours.

### Results

| Algorithm     | TOP1 Loss    | **TOP2** Loss | **TOP3 **Loss | **TOP4 **Loss | **TOP5 **Loss | **TOP6** Loss | **TOP7 **Loss | **TOP8 **Loss | **TOP9 **Loss | **TOP10 **Loss |
| ------------- | ------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------- |
| Random Search | 0.418854     | 0.419595      | 0.419945      | 0.421683      | 0.421685      | 0.422116      | 0.422528      | 0.422983      | 0.423035      | 0.423108       |
| Random Search | 0.417364     | 0.419173      | 0.420747      | 0.421403      | 0.421433      | 0.421437      | 0.421759      | 0.421788      | 0.422371      | 0.422493       |
| Random Search | 0.417861     | 0.41816       | 0.420704      | 0.420736      | 0.421258      | 0.421312      | 0.421417      | 0.421427      | 0.421469      | 0.422079       |
| Grid Search   | 0.498166     | 0.498166      | 0.498166      | 0.498166      | 0.498166      | 0.498166      | 0.498167      | 0.498167      | 0.498167      | 0.498167       |
| Evolution     | 0.409887     | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887       |
| Evolution     | 0.413620     | 0.413620      | 0.413620      | 0.414258      | 0.414258      | 0.414258      | 0.414258      | 0.414258      | 0.414258      | 0.414258       |
| Evolution     | 0.409887     | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887       |
| Anneal        | 0.414877     | 0.416751      | 0.417743      | 0.418002      | 0.419073      | 0.419225      | 0.419285      | 0.419285      | 0.419285      | 0.419285       |
| Anneal        | 0.409887     | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.409887      | 0.410464      | 0.410464      | 0.410464      | 0.410464       |
| Anneal        | 0.413683     | 0.417766      | 0.417766      | 0.417766      | 0.417766      | 0.417766      | 0.417766      | 0.417766      | 0.418663      | 0.418663       |
| Metis         | 0.416273     | 0.419946      | 0.420192      | 0.422823      | 0.422823      | 0.423395      | 0.423898      | 0.424424      | 0.424950      | 0.425076       |
| Metis         | 0.420262     | 0.422190      | 0.423942      | 0.424236      | 0.425243      | 0.425919      | 0.426157      | 0.426588      | 0.426690      | 0.426931       |
| Metis         | 0.421027     | 0.424222      | 0.424589      | 0.425349      | 0.425675      | 0.426296      | 0.427059      | 0.427166      | 0.427427      | 0.428334       |
| TPE           | 0.414478     | 0.414478      | 0.414478      | 0.414478      | 0.414478      | 0.414478      | 0.414478      | 0.414478      | 0.414478      | 0.414478       |
| TPE           | 0.415077     | 0.416594      | 0.419297      | 0.419480      | 0.419480      | 0.419480      | 0.419480      | 0.419480      | 0.419480      | 0.420124       |
| TPE           | 0.415077     | 0.417007      | 0.417125      | 0.417677      | 0.418157      | 0.418295      | 0.418556      | 0.419201      | 0.419717      | 0.419717       |
| SMAC          | **0.408386** | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386       |
| SMAC          | 0.414012     | 0.414012      | 0.414012      | 0.414012      | 0.414012      | 0.414012      | 0.414012      | 0.414012      | 0.414012      | 0.414012       |
| SMAC          | **0.408386** | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386      | 0.408386       |
| BOHB          | 0.410464     | 0.411563      | 0.415848      | 0.41936       | 0.41936       | 0.419557      | 0.419946      | 0.420449      | 0.420449      | 0.420549       |
| BOHB          | 0.418995     | 0.418995      | 0.420547      | 0.420692      | 0.42211       | 0.42211       | 0.425604      | 0.425664      | 0.425664      | 0.425664       |
| BOHB          | 0.415149     | 0.418463      | 0.418568      | 0.419091      | 0.419091      | 0.419385      | 0.419654      | 0.419654      | 0.420133      | 0.420133       |
| HyperBand     | 0.414065     | 0.414065      | 0.414065      | 0.416957      | 0.416957      | 0.419717      | 0.419717      | 0.420001      | 0.420001      | 0.420735       |
| HyperBand     | 0.416807     | 0.416807      | 0.416807      | 0.418662      | 0.418662      | 0.419305      | 0.419305      | 0.419305      | 0.420806      | 0.421810       |
| HyperBand     | 0.415550     | 0.415859      | 0.415859      | 0.415859      | 0.416757      | 0.418090      | 0.418090      | 0.418245      | 0.418245      | 0.419304       |

For Metis, there are about 300 trials because it runs slowly due to its high time complexity O(n^3) in Gaussian Process.

## RocksDB Benchmark 'fillrandom' and 'readrandom'

### Problem Description

`DB_Bench` is the main tool that is used to benchmark RocksDB's performance. It has so many hapermeter to tune.

The performance of `DB_Bench` is associated with the machine configuration and installation method. We run the `DB_Bench`in the Linux machine and install the Rock in shared library.

#### Machine configuration

```
RocksDB:    version 6.1
CPU:        6 * Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
CPUCache:   35840 KB
Keys:       16 bytes each
Values:     100 bytes each (50 bytes after compression)
Entries:    1000000
```

#### Storage performance

**Latency**: each IO request will take some time to complete, this is called the average latency. There are several factors that would affect this time including network connection quality and hard disk IO performance.

**IOPS**: **IO operations per second**, which means the amount of _read or write operations_ that could be done in one seconds time.

**IO size**: **the size of each IO request**. Depending on the operating system and the application/service that needs disk access it will issue a request to read or write a certain amount of data at the same time.

**Throughput (in MB/s) = Average IO size x IOPS **

IOPS is related to online processing ability and we use the IOPS as the metric in my experiment.

### Search Space

```json
{
  "max_background_compactions": {
    "_type": "quniform",
    "_value": [1, 256, 1]
  },
  "block_size": {
    "_type": "quniform",
    "_value": [1, 500000, 1]
  },
  "write_buffer_size": {
    "_type": "quniform",
    "_value": [1, 130000000, 1]
  },
  "max_write_buffer_number": {
    "_type": "quniform",
    "_value": [1, 128, 1]
  },
  "min_write_buffer_number_to_merge": {
    "_type": "quniform",
    "_value": [1, 32, 1]
  },
  "level0_file_num_compaction_trigger": {
    "_type": "quniform",
    "_value": [1, 256, 1]
  },
  "level0_slowdown_writes_trigger": {
    "_type": "quniform",
    "_value": [1, 1024, 1]
  },
  "level0_stop_writes_trigger": {
    "_type": "quniform",
    "_value": [1, 1024, 1]
  },
  "cache_size": {
    "_type": "quniform",
    "_value": [1, 30000000, 1]
  },
  "compaction_readahead_size": {
    "_type": "quniform",
    "_value": [1, 30000000, 1]
  },
  "new_table_reader_for_compaction_inputs": {
    "_type": "randint",
    "_value": [1]
  }
}
```

The search space is enormous (about 10^40) and we set the maximum number of trial to 100 to limit the computation resource.

### Results

#### fillrandom' Benchmark

| Model     | Best IOPS (Repeat 1) | Best IOPS (Repeat 2) | Best IOPS (Repeat 3) |
| --------- | -------------------- | -------------------- | -------------------- |
| Random    | 449901               | 427620               | 477174               |
| Anneal    | 461896               | 467150               | 437528               |
| Evolution | 436755               | 389956               | 389790               |
| TPE       | 378346               | 482316               | 468989               |
| SMAC      | 491067               | 490472               | **491136**           |
| Metis     | 444920               | 457060               | 454438               |

Figure:

![](../img/hpo_rocksdb_fillrandom.PNG)

#### 'readrandom' Benchmark

| Model     | Best IOPS (Repeat 1) | Best IOPS (Repeat 2) | Best IOPS (Repeat 3) |
| --------- | -------------------- | -------------------- | -------------------- |
| Random    | 2276157              | 2285301              | 2275142              |
| Anneal    | 2286330              | 2282229              | 2284012              |
| Evolution | 2286524              | 2283673              | 2283558              |
| TPE       | 2287366              | 2282865              | 2281891              |
| SMAC      | 2270874              | 2284904              | 2282266              |
| Metis     | **2287696**          | 2283496              | 2277701              |

Figure:

![](../img/hpo_rocksdb_readrandom.PNG)