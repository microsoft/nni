"""Hack autodoc to get more fine-grained docstring rendering contol."""

import inspect
import os
from typing import List, Any

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


def trial_tool_import_patch(*args, **kwargs):
    """Insert dummy trial tool variable to ensure trial_tool can be imported."""
    os.environ.update({
        'NNI_OUTPUT_DIR': '/tmp',
        'NNI_PLATFORM': 'unittest',
        'NNI_SYS_DIR': '/tmp',
        'NNI_TRIAL_JOB_ID': 'dummy',
        'NNI_EXP_ID': 'dummy',
        'MULTI_PHASE': 'dummy'
    })


class FindAutosummaryFilesPatch:
    """Ignore certain files as they are completely un-importable."""

    original = None

    blacklist = [
        'nni.retiarii.codegen.tensorflow',
        'nni.nas.benchmarks.nasbench101.db_gen',
        'nni.tools.jupyter_extension.management',
    ]

    def restore(self, *args, **kwargs):
        assert self.original is not None
        sphinx.ext.autosummary.generate.find_autosummary_in_files = self.original

    def patch(self, app, config):
        from sphinx.ext.autosummary.generate import AutosummaryEntry

        self.original = sphinx.ext.autosummary.generate.find_autosummary_in_files

        def find_autosummary_in_files(filenames: List[str]) -> List[AutosummaryEntry]:
            items: List[AutosummaryEntry] = self.original(filenames)
            items = [item for item in items if item.name not in config.autosummary_mock_imports]
            return items

        sphinx.ext.autosummary.generate.find_autosummary_in_files = find_autosummary_in_files


def setup(app):
    # See life-cycle of sphinx app here:
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    patch = ClassNewBlacklistPatch()
    app.connect('env-before-read-docs', patch.patch)
    app.connect('env-merge-info', patch.restore)

    app.connect('env-before-read-docs', disable_trace_patch)

    # autosummary generate happens at builder-inited
    app.connect('config-inited', trial_tool_import_patch)

    autosummary_patch = FindAutosummaryFilesPatch()
    app.connect('config-inited', autosummary_patch.patch)
    app.connect('env-merge-info', autosummary_patch.restore)
