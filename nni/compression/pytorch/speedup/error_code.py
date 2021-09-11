# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# Error Code of the speedup
class SpeedupError(Exception):
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return str(self.msg)

class EmptyLayerError(SpeedupError):
    def __init__(self):
        super(EmptyLayerError, self).__init__("Pruning a Layer to empty is not legal")

class ShapeMisMatchError(SpeedupError):
    def __init__(self):
        super(ShapeMisMatchError, self).__init__("Shape mismatch!")

class InputsNumberError(SpeedupError):
    def __init__(self):
        super(InputsNumberError, self).__init__("The number of the inputs of the target OP is wrong")

class OutputTypeError(SpeedupError):
    def __init__(self, current_type, target_type):
        msg = f"The output type should be {str(target_type)}, but {str(current_type)} founded"
        super(OutputTypeError, self).__init__(msg)

class UnBalancedGroupError(SpeedupError):
    def __init__(self):
        msg = "The number remained filters in each group is different"
        super(UnBalancedGroupError, self).__init__(msg)