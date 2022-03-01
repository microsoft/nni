"""Hack autodoc to get more fine-grained docstring rendering contol."""

import inspect
import os

import sphinx


class ClassNewBlacklistPatch:
    """Force some classes to skip ``__new__`` when generating signature."""

    original = None

    def restore(self, *args, **kwargs):
        assert self.original is not None
        sphinx.ext.autodoc._CLASS_NEW_BLACKLIST = self.original

    def patch(self, *args, **kwargs):
        self.original = sphinx.ext.autodoc._CLASS_NEW_BLACKLIST

        blacklist = []

        import nni.retiarii.nn.pytorch
        for name in dir(nni.retiarii.nn.pytorch):
            obj = getattr(nni.retiarii.nn.pytorch, name)
            if inspect.isclass(obj):
                new_name = "{0.__module__}.{0.__qualname__}".format(obj.__new__)
                if new_name not in blacklist:
                    blacklist.append(new_name)

        sphinx.ext.autodoc._CLASS_NEW_BLACKLIST = self.original + blacklist


def disable_trace_patch(*args, **kwargs):
    """Disable trace by setting an environment variable."""
    os.environ['NNI_TRACE_FLAG'] = 'DISABLE'



def setup(app):
    # See life-cycle of sphinx app here:
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    patch = ClassNewBlacklistPatch()
    app.connect('env-before-read-docs', patch.patch)
    app.connect('env-merge-info', patch.restore)

    app.connect('env-before-read-docs', disable_trace_patch)
