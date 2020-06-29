authorName: default
experimentName: example_mnist
trialConcurrency: 1
maxExecDuration: 1h
maxTrialNum: 10
#choice: local, remote, pai
trainingServicePlatform: local
#please use `nnictl ss_gen` to generate search space file first
searchSpacePath: <the_generated_search_space_path>
useAnnotation: False
tuner:
  codeDir: ../../tuners/random_nas_tuner
  classFileName: random_nas_tuner.py
  className: RandomNASTuner
trial:
  command: python3 mnist.py
  codeDir: .
  gpuNum: 0
