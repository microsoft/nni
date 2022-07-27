# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base import BaseStrategy

try:
    from nni.retiarii.oneshot.pytorch.strategy import (  # pylint: disable=unused-import
        DARTS, GumbelDARTS, Proxyless, ENAS, RandomOneShot
    )
except ImportError as import_err:
    _import_err = import_err

    class ImportFailedStrategy(BaseStrategy):
        def run(self, base_model, applied_mutators):
            raise _import_err

    # otherwise typing check will pointing to the wrong location
    globals()['DARTS'] = ImportFailedStrategy
    globals()['GumbelDARTS'] = ImportFailedStrategy
    globals()['Proxyless'] = ImportFailedStrategy
    globals()['ENAS'] = ImportFailedStrategy
    globals()['RandomOneShot'] = ImportFailedStrategy
