## How to download data

1. download "dev-v1.1.json" and "train-v1.1.json" in https://rajpurkar.github.io/SQuAD-explorer/
2. download "glove.840B.300d.txt" in "https://nlp.stanford.edu/projects/glove/"

## How to submit this job

1. run "$NNI_ROOT_DIR/auto_run.py" as "$NNI_ROOT_DIR/README-AUTO.md" said.
2. use the dockerImage openpai.azurecr.io/nni_v0.0.1, which means it use a tensorflow cpu-version.
3. this model don't need search_space.json.