# Download data

## Use downloading script

Execute the following command to download needed files
using the downloading script:

```
chmod +x ./download.sh
./download.sh
```

## Download manually

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

# How to submit this job

1. run "$NNI_ROOT_DIR/auto_run.py" as "$NNI_ROOT_DIR/README-AUTO.md" said.
2. use the dockerImage openpai.azurecr.io/nni_v0.0.1, which means it use a tensorflow cpu-version.
3. this model don't need search_space.json.