from network import ShuffleNetV2OneShot
model = ShuffleNetV2OneShot()

from nni.retiarii.oneshot.pytorch.random import HardwareLatencyEstimator
predictor = 'cortexA76cpu_tflite21'
HardwareLatencyEstimator(predictor, model)
