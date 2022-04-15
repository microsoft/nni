import tensorrt as trt
import common
import time
import pycuda.driver as cuda
import torch
import os

TRT_LOGGER = trt.Logger()

class CalibrateType:
    LEGACY = trt.CalibrationAlgoType.LEGACY_CALIBRATION
    ENTROPY = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
    ENTROPY2 = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    MINMAX = trt.CalibrationAlgoType.MINMAX_CALIBRATION

Precision_Dict = {
    8: trt.int8,
    16: trt.float16,
    32: trt.float32
}


class BertCalibrator(trt.IInt8Calibrator):
    def __init__(self, training_data, cache_file="hubert.cache", batch_size=64, algorithm=trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2):
        trt.IInt8Calibrator.__init__(self)

        self.algorithm = algorithm
        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.data = training_data
        self.batch_size = batch_size
        self.current_index = 0

        # Allocate enough memory for a whole batch.
        self.device_input = [cuda.mem_alloc(subdata.nbytes) for subdata in self.data]
        # self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index == 1:
            print("Calibration successfully")
            return None
        cuda.memcpy_htod(self.device_input[0], self.data[0].ravel())

        self.current_index += 1
        return self.device_input


    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def inference(context, test_data):
    inputs, outputs, bindings, stream = common.allocate_buffers(context.engine)
    result = []
    inputs[0].host = test_data

    _, elapsed_time = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)

    return result, elapsed_time

# This function builds an engine from a Onnx model.
def build_engine(model_file, calib, batch_size=32):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser, builder.create_builder_config() as trt_config:

        # Attention that, builder should be set to 1 because of the implementation of allocate_buffer
        builder.max_batch_size = 1
        # builder.max_workspace_size = common.GiB(1)
        trt_config.max_workspace_size = common.GiB(4)


        #builder.int8_mode = True
        #builder.fp16_mode = True
        trt_config.set_flag(trt.BuilderFlag.INT8)
        trt_config.set_flag(trt.BuilderFlag.FP16)

        #builder.int8_calibrator = calib
        trt_config.int8_calibrator = calib

        
        # Parse onnx model
        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None


        # This design may not be correct if output more than one

        """
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            layer.precision = trt.int8
            layer.set_output_type(0, trt.int8)
        """
        

        # network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        # engine = builder.build_cuda_engine(network)
        engine = builder.build_engine(network, trt_config)
        print("engine build successfully")
        return engine

#onnx_path = "/data/znx/hubert/transformers/examples/pytorch/audio-classification/hubert_new.onnx"
#onnx_path = "/data/znx/hubert/transformers/examples/pytorch/audio-classification/hubert_ori_onnx/hubert_ori.onnx"
onnx_path = "../../checkpoints/hubert/artifact_hubert_coarse_no_propagation_onnx_with_tesa/model_no_tesa.onnx"
dummy_input = (torch.rand(32, 16000).numpy())

calib = BertCalibrator(dummy_input)
engine = build_engine(onnx_path, calib)
print("Engine generate successfully")
context = engine.create_execution_context()

time_set = []
for i in range(100):
    _, time = inference(context, dummy_input)
    time_set.append(time)

print(f'average time: {sum(time_set)/len(time_set)* 1000} ms')
