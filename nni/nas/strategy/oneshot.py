# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base import Strategy

try:
    from nni.nas.oneshot.pytorch.strategy import (  # pylint: disable=unused-import
        DARTS, GumbelDARTS, Proxyless, ENAS, RandomOneShot
    )
except ImportError as import_err:
    _import_err = import_err

    class ImportFailedStrategy(Strategy):
        def run(self, *args, **kwargs):
            raise _import_err

    # otherwise typing check will pointing to the wrong location
    globals()['DARTS'] = ImportFailedStrategy
    globals()['GumbelDARTS'] = ImportFailedStrategy
    globals()['Proxyless'] = ImportFailedStrategy
    globals()['ENAS'] = ImportFailedStrategy
    globals()['RandomOneShot'] = ImportFailedStrategy
