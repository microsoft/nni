# Copyright (c) Microsoft Corporation. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

from abc import abstractmethod

from sklearn.metrics import accuracy_score, mean_squared_error


class Metric:
    @classmethod
    @abstractmethod
    def higher_better(cls):
        pass

    @classmethod
    @abstractmethod
    def compute(cls, prediction, target):
        pass

    @classmethod
    @abstractmethod
    def evaluate(cls, prediction, target):
        pass


class Accuracy(Metric):
    @classmethod
    def higher_better(cls):
        return True

    @classmethod
    def compute(cls, prediction, target):
        prediction = list(map(lambda x: x.argmax(), prediction))
        target = list(map(lambda x: x.argmax(), target))
        return cls.evaluate(prediction, target)

    @classmethod
    def evaluate(cls, prediction, target):
        return accuracy_score(prediction, target)


class MSE(Metric):
    @classmethod
    def higher_better(cls):
        return False

    @classmethod
    def compute(cls, prediction, target):
        return cls.evaluate(prediction, target)

    @classmethod
    def evaluate(cls, prediction, target):
        return mean_squared_error(prediction, target)
