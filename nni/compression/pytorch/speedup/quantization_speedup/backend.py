import nni.compression.pytorch.speedup.quantization_speedup.integrated_tensorrt as integrated_tensorrt
from nni.compression.pytorch.speedup.quantization_speedup.integrated_tensorrt import CalibrateType

class BackendEngine:
    def __init__(self, backend, model, onnx_path, input_shape, config=None, extra_layer_bit=32, 
        strict_datatype=False, using_calibrate=True, calibrate_type=None, calib_data=None, 
        calibration_cache = None, batchsize=1, input_names=["actual_input_1"], output_names=["output1"]):
        """
        Parameters
        ----------
        backend : str
            The backend user want to run inference such as tensorrt, nnfusion, tvm, etc.
            Only support tensorrt right now.
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
        self.backend = backend
        if self.backend == 'tensorrt':
            self.engine = integrated_tensorrt.TensorRt(model, onnx_path, input_shape, config=config, 
                extra_layer_bit=32, strict_datatype=True, using_calibrate=True, calibrate_type=CalibrateType.ENTROPY2, 
                calib_data=calib_data, calibration_cache = calibration_cache, batchsize=batchsize, 
                input_names=input_names, output_names=output_names)
        else:
            raise NameError('Only support tensorrt as backend now.')
            # self.engine = NewBackend(*args)
    
    def build(self):
        if self.backend == 'tensorrt':
            self.engine.tensorrt_build()
        else:
            raise NameError('Only support tensorrt as backend now.')
            # self.engine.newengine_build()
    
    def inference(self, test_set=None):
        assert test_set is not None
        output, time = self.engine.inference(test_set)
        return output, time
