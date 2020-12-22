from onnx.onnx_ONNX_REL_1_8_ml_pb2 import ModelProto
import tensorrt as trt
import frontend_to_onnx as fonnx
import calibrator
import common
import time

TRT_LOGGER = trt.Logger()

class CalibrateType:
    LEGACY = trt.CalibrationAlgoType.LEGACY_CALIBRATION
    ENTROPY = trt.CalibrationAlgoType.ENTROPY_CALIBRATION
    ENTROPY2 = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
    MINMAX = trt.CalibrationAlgoType.MINMAX_CALIBRATION

class Precision:
    trt8 = trt.int8
    trt16 = trt.float16
    trti32 = trt.int32
    trt32 = trt.float32

Precision_Dict = {
    "int8": trt.int8,
    "float16": trt.float16,
    "int32": trt.int32,
    "float32": trt.float32
}

# This function builds an engine from a Onnx model.
def build_engine(model_file, calib, batch_size=32, config=None, extra_layer_bit='float32', strict_datatype=False):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Attention that, builder should be set to 1 because of the implementation of allocate_buffer
        builder.max_batch_size = 1
        builder.max_workspace_size = common.GiB(1)

        if extra_layer_bit is "float32" and config is None:
            pass
        elif extra_layer_bit is "int8" and config is None:
            builder.int8_mode = True
            builder.int8_calibrator = calib
        else:
            builder.int8_mode = True
            builder.fp16_mode = True
            builder.int8_calibrator = calib
            builder.strict_type_constraints = strict_datatype
        
        # Parse onnx model

        with open(model_file, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None
        
        assert extra_layer_bit in Precision_Dict
        extra_datatype = Precision_Dict[extra_layer_bit]

        # This design may not be correct if output more than one
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            if layer.name in config:
                bitset = config[layer.name]
                layer.precision = Precision_Dict[bitset]
                layer.set_output_type(0, Precision_Dict[bitset])
            else:
                layer.precision = extra_datatype
                layer.set_output_type(0, extra_datatype)
        # network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        engine = builder.build_cuda_engine(network)
        return engine

class TensorRt:
    def __init__(self, model, onnx_path, input_shape, config=None, extra_layer_bit='float32', strict_datatype=False, using_calibrate=True, 
    calibrate_type=None, calib_data=None, calibration_cache = None, batchsize=1, input_names=["actual_input_1"], output_names=["output1"]):
        self.model = model
        self.onnx_path = onnx_path
        self.input_shape = input_shape
        self.config = config
        self.extra_layer_bit = extra_layer_bit
        self.strict_datatype = strict_datatype
        self.using_calibrate = using_calibrate
        self.calibrate_type = calibrate_type
        self.calib_data = calib_data
        self.calibration_cache = calibration_cache
        self.batchsize = batchsize
        self.input_names = input_names
        self.output_names = output_names
        self.context = None

    def tensorrt_build(self):
        assert self.model is not None
        assert self.onnx_path is not None
        assert self.input_shape is not None

        # convert pytorch model to onnx model and save onnx model in onnx_path
        model_onnx = fonnx.torch_to_onnx(self.model, input_shape=self.input_shape, model_path=self.onnx_path, input_names=self.input_names, output_names=self.output_names)

        if self.using_calibrate:
            assert self.calibrate_type is not None
            context = self.tensorrt_build_withcalib(self.onnx_path)
        else:
            try:
                raise NameError('quantized without calibrate has not been supported.')
            except NameError:
                raise
        self.context = context

    def tensorrt_build_withcalib(self, onnx_path):
        calib = calibrator.Calibrator(self.calib_data, self.calibration_cache, self.batchsize, self.calibrate_type)
        engine = build_engine(onnx_path, calib, self.batchsize, self.config, self.extra_layer_bit, self.strict_datatype)
        return engine.create_execution_context()

    def inference(self, test_data):
        elapsed_time = 0
        inputs, outputs, bindings, stream = common.allocate_buffers(self.context.engine)
        result = []
        for start_idx in range(0, test_data.shape[0], self.batchsize):
            # If the number of images in the test set is not divisible by the batch size, the last batch will be smaller.
            # This logic is used for handling that case.
            end_idx = min(start_idx + self.batchsize, test_data.shape[0])
            effective_batch_size = end_idx - start_idx

            # Do inference for every batch.
            inputs[0].host = test_data[start_idx:start_idx + effective_batch_size]
            t1 = time.time()
            [output] = common.do_inference_v2(self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            elapsed_time += time.time() - t1
            result.append(output.copy())
            # Use argmax to get predictions and then check accuracy
        return result, elapsed_time

