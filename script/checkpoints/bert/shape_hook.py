import torch
import torchvision
import json
class ShapeHook:
    def __init__(self, model, dummy_input, debug_point=None):
        assert isinstance(model, torch.nn.Module)
        self.model = model
        model.eval()
        self.dummy_input = dummy_input
        self.shapes = {}
        for name, module in self.model.named_modules():
            self.shapes[name] = {"in_shape":[], "out_shape":[], "weight_shape":[], "type":None}
        self.trace_points = []
        if debug_point != None:
            self.trace_points = debug_point
        self.__deploy_hooks()
        if isinstance(dummy_input, torch.Tensor):
            self.model(dummy_input)
        elif isinstance(dummy_input, tuple) or isinstance(dummy_input, list):
            self.model(*dummy_input)
        elif isinstance(dummy_input, dict):
            self.model(**dummy_input)

    def __parse_tensor_shapa(self, args):
        shapes = []
        if isinstance(args, dict):
            # some layers may return their output as a dict 
            # ex. the IntermediateLayerGetter in the face detection jobs.
            for key, val in args.items():
                if isinstance(val, torch.Tensor):
                    shapes.append(list(val.size())+[str(val.dtype)])
                else:
                    shapes.extend(self.__parse_tensor_shapa(val))
        elif isinstance(args, list) or isinstance(args, tuple):
            # list or tuple
            for item in args:
                if isinstance(item, torch.Tensor):
                    shapes.append(list(item.size()) + [str(item.dtype)])
                else:
                    shapes.extend(self.__parse_tensor_shapa(item))
        elif isinstance(args, torch.Tensor) or isinstance(args, torch.autograd.Variable):
            # if the output is a already a tensor/variable, then return itself
            shapes.append(list(args.size()) + [str(args.dtype)])
        return shapes

    def __get_decorator(self, func, name, debug_on=False):
        def new_func(*args, **kwargs):
            if debug_on:
                # 
                import pdb; pdb.set_trace()
            self.shapes[name]['in_shape'].extend(self.__parse_tensor_shapa(args))
            self.shapes[name]['in_shape'].extend(self.__parse_tensor_shapa(kwargs))
            out = func(*args, **kwargs)
            self.shapes[name]['out_shape'].extend(self.__parse_tensor_shapa(out))
            return out
        return new_func

    def __deploy_hooks(self):
        """
        Deploy the hooks to get the input/output/weight shape of all submodules.
        """
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
                self.shapes[name]['weight_shape'].append(module.weight.size())
            self.shapes[name]['type'] = str(type(module))
            setattr(module, 'forward', self.__get_decorator(module.forward, name, name in self.trace_points))

    def export(self, fpath):
        with open(fpath, 'w') as f:
            json.dump(self.shapes, f, indent=4)


if __name__ == '__main__':
    from cifar_models.mobilenetv2 import MobileNetV2
    from cifar_models.resnet import ResNet50
    from cifar_models.mlp_mnist import MLP
    batch_size = 32
    dummy_input = torch.rand(batch_size, 3, 32, 32)
    hook_1 = ShapeHook(MobileNetV2(), dummy_input)
    hook_2 = ShapeHook(ResNet50(), dummy_input)
    hook_3 = ShapeHook(MLP(), torch.rand(batch_size, 1 , 32, 32)) # Mnist
    hook_1.export('mobilenetv2_shape.json')
    hook_2.export('resnet50_shape.json')
    hook_3.export('mlp_mnist_shape.json')
    