# Automatic Model Architecture Search for Reading Comprehension
This example shows us how to use Genetic Algorithm to find good model architectures for Reading Comprehension task.

## Search Space
Since attention and recurrent neural network (RNN) module have been proven effective in Reading Comprehension.
We conclude the search space as follow:

1. IDENTITY (Effectively means keep training).
2. INSERT-RNN-LAYER (Inserts a LSTM. Comparing the performance of GRU and LSTM in our experiment, we decided to use LSTM here.)
3. REMOVE-RNN-LAYER
4. INSERT-ATTENTION-LAYER(Inserts a attention layer.)
5. REMOVE-ATTENTION-LAYER
6. ADD-SKIP (Identity between random layers).
7. REMOVE-SKIP (Removes random skip).

![ga-squad-logo](./ga_squad.png)

## New version
Also we have another version which time cost is less and performance is better. We will release soon.

# How to run this example?

## Download data

### Use downloading script to download data

Execute the following command to download needed files
using the downloading script:

```
chmod +x ./download.sh
./download.sh
```

### Download manually

1. download "dev-v1.1.json" and "train-v1.1.json" in https://rajpurkar.github.io/SQuAD-explorer/

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

2. download "glove.840B.300d.txt" in https://nlp.stanford.edu/projects/glove/

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip
```

## Update configuration
Modify `nni/examples/trials/ga_squad/config.yaml`, here is the default configuration:

```
authorName: default
experimentName: example_ga_squad
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 1
#choice: local, remote
trainingServicePlatform: local
#choice: true, false
useAnnotation: false
tuner:
  codeDir: ~/nni/examples/tuners/ga_customer_tuner
  classFileName: customer_tuner.py
  className: CustomerTuner
  classArgs:
    optimize_mode: maximize
trial:
  command: python3 trial.py
  codeDir: ~/nni/examples/trials/ga_squad
  gpuNum: 0
```

In the "trial" part, if you want to use GPU to perform the architecture search, change `gpuNum` from `0` to `1`. You need to increase the `maxTrialNum` and `maxExecDuration`, according to how long you want to wait for the search result.

`trialConcurrency` is the number of trials running concurrently, which is the number of GPUs you want to use, if you are setting `gpuNum` to 1.

## submit this job

```
nnictl create --config ~/nni/examples/trials/ga_squad/config.yaml
```

# Techinal details about the trial

## Model configuration format

Here is an example of the model configuration, which is passed from the tuner to the trial in the architecture search procedure.

```
{
    "max_layer_num": 50,
    "layers": [
        {
            "input_size": 0,
            "type": 3,
            "output_size": 1,
            "input": [],
            "size": "x",
            "output": [4, 5],
            "is_delete": false
        },
        {
            "input_size": 0,
            "type": 3,
            "output_size": 1,
            "input": [],
            "size": "y",
            "output": [4, 5],
            "is_delete": false
        },
        {
            "input_size": 1,
            "type": 4,
            "output_size": 0,
            "input": [6],
            "size": "x",
            "output": [],
            "is_delete": false
        },
        {
            "input_size": 1,
            "type": 4,
            "output_size": 0,
            "input": [5],
            "size": "y",
            "output": [],
            "is_delete": false
        },
        {"Comment": "More layers will be here for actual graphs."}
    ]
}
```

Every model configuration will has a "layers" section, which is a JSON list of layer definitions. The definition of each layer is also a JSON object, where:

 * `type` is the type of the layer. 0, 1, 2, 3, 4 corresponde to attention, self-attention, RNN, input and output layer respectively.
 * `size` is the length of the output. "x", "y" corresponde to document length / question length, respectively.
 * `input_size` is the number of inputs the layer has.
 * `input` is the indices of layers taken as input of this layer.
 * `output` is the indices of layers use this layer's output as their input.
 * `is_delete` means whether the layer is still available.