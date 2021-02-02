import tensorrt as trt
import nni.compression.pytorch.speedup.quantization_speedup.frontend_to_onnx as fonnx
import nni.compression.pytorch.speedup.quantization_speedup.calibrator as calibrator
import nni.compression.pytorch.speedup.quantization_speedup.common as common
import time

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

def build_engine(model_file, calib, batch_size=32, config=None, extra_layer_bit='float32', strict_datatype=False):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        """
        This function builds an engine from an onnx model.
        """
        # Attention that, builder should be set to 1 because of the implementation of allocate_buffer
        builder.max_batch_size = 1
        builder.max_workspace_size = common.GiB(1)

        if extra_layer_bit == 32 and config is None:
            pass
        elif extra_layer_bit == 8 and config is None:
            # entire model in 8bit mode
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
                print ('ERROR: Fail to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        # This design may not be correct if output more than one
        for i in range(network.num_layers):
            if config is None:
                break
            layer = network.get_layer(i)
            if layer.name in config:
                bitset = config[layer.name]
                layer.precision = Precision_Dict[bitset]
                layer.set_output_type(0, Precision_Dict[bitset])
        # network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        engine = builder.build_cuda_engine(network)
        return engine

class TensorRt:
    def __init__(self, model, onnx_path, input_shape, config=None, extra_layer_bit=32, strict_datatype=False, using_calibrate=True, 
    calibrate_type=None, calib_data=None, calibration_cache = None, batchsize=1, input_names=["actual_input_1"], output_names=["output1"]):
        """
        Parameters
        ----------
        model : pytorch model
            The model to speed up by quantization.
        onnx_path : str
            The path user want to store onnx model which is converted from pytorch model.
        input_shape : tuple
            The input shape of model, shall pass it to torch.onnx.export.
        config : dict
            Config recording bit number and name of layers.
        extra_layer_bit : int
            Other layers which are not in config will be quantized to corresponding bit number.
        strict_datatype : bool
            Whether constrain layer bit to the number given in config or not. If true, all the layer 
            will be set to given bit strictly. Otherwise, these layers will be set automatically by
            tensorrt.
        using_calibrate : bool
            Whether calibrating during quantization or not. If true, user should provide calibration
            dataset. If not, user should provide scale and zero_point for each layer. Current version
            only support using calibrating.
        calibrate_type : tensorrt.tensorrt.CalibrationAlgoType
            The algorithm of calibrating. Please refer to https://docs.nvidia.com/deeplearning/
            tensorrt/api/python_api/infer/Int8/Calibrator.html for detail
        calibrate_data : numpy array
            The data using to calibrate quantization model
        calibration_cache : str
            The path user want to store calibrate cache file
        batchsize : int
            The batch size of calibration and inference
        input_names : list
            Input name of onnx model providing for torch.onnx.export to generate onnx model
        output_name : list
            Output name of onnx model providing for torch.onnx.export to generate onnx model
        """
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
        self.onnx_config = {}

    def tensorrt_build(self):
        """
        Get onnx config and build tensorrt engine.
        """
        assert self.model is not None
        assert self.onnx_path is not None
        assert self.input_shape is not None

        # Convert pytorch model to onnx model and save onnx model in onnx_path
        _, self.onnx_config = fonnx.torch_to_onnx(self.model, self.config, input_shape=self.input_shape, model_path=self.onnx_path, input_names=self.input_names, output_names=self.output_names)

        if self.using_calibrate:
            assert self.calibrate_type is not None
            context = self.tensorrt_build_withcalib(self.onnx_path)
        else:
            raise NameError('quantized without calibrate has not been supported.')
        self.context = context

    def tensorrt_build_withcalib(self, onnx_path):
        calib = calibrator.Calibrator(self.calib_data, self.calibration_cache, self.batchsize, self.calibrate_type)
        engine = build_engine(onnx_path, calib, self.batchsize, self.onnx_config, self.extra_layer_bit, self.strict_datatype)
        return engine.create_execution_context()

    def inference(self, test_data):
        """
        Do inference by tensorrt builded engine.
        """
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
            shape = output.shape[0]
            output = output[0:int(shape * effective_batch_size / self.batchsize)]
            elapsed_time += time.time() - t1
            result.append(output.copy())
            # Use argmax to get predictions and then check accuracy
        return result, elapsed_time