"""Hack autodoc to get more fine-grained docstring rendering contol.

autodoc and autosummary didn't expose many of their controls to sphinx users via config.
To customize them, the "correct" approach seems to copy and paste all their code and rewrite some part.
To avoid doing this, I monkey-patched some of the functions to keep the changes minimal.

Note that some of them are related to sphinx internal APIs, which can be broken when sphinx got upgraded.
Try to keep them updated, or pin to a particular sphinx version.
"""

import inspect
import os
from typing import List, Tuple, List

import sphinx
from docutils import nodes
from docutils.nodes import Node


class ClassNewBlacklistPatch:
    """Force some classes to skip ``__new__`` when generating signature."""

    original = None

    def restore(self, *args, **kwargs):
        assert self.original is not None
        sphinx.ext.autodoc._CLASS_NEW_BLACKLIST = self.original

    def patch(self, *args, **kwargs):
        self.original = sphinx.ext.autodoc._CLASS_NEW_BLACKLIST

        blacklist = []

        # import nni.retiarii.nn.pytorch
        # for name in dir(nni.retiarii.nn.pytorch):
        #     obj = getattr(nni.retiarii.nn.pytorch, name)
        #     if inspect.isclass(obj):
        #         new_name = "{0.__module__}.{0.__qualname__}".format(obj.__new__)
        #         if new_name not in blacklist:
        #             blacklist.append(new_name)

        sphinx.ext.autodoc._CLASS_NEW_BLACKLIST = self.original + blacklist


def disable_trace_patch(*args, **kwargs):
    """Disable trace by setting an environment variable."""
    os.environ['NNI_TRACE_FLAG'] = 'DISABLE'


def trial_tool_import_patch(*args, **kwargs):
    """Insert dummy trial tool variable to ensure trial_tool can be imported.
    See nni/tools/trial_tool/constants.py
    """
    os.environ.update({
        'NNI_OUTPUT_DIR': '/tmp',
        'NNI_PLATFORM': 'unittest',
        'NNI_SYS_DIR': '/tmp',
        'NNI_TRIAL_JOB_ID': 'dummy',
        'NNI_EXP_ID': 'dummy',
        'MULTI_PHASE': 'dummy'
    })


class AutoSummaryPatch:
    """Ignore certain files as they are completely un-importable. It patches:

    - find_autosummary_in_files: Some modules cannot be imported at all due to dependency issues or some special design.
      They need to skipped when running autosummary generate.
    - Autosummary.get_table: The original autosummary creates an index for each module, and the module links in autosummary table
      points to the corresponding generated module page (by using ``:py:module:xxx``). This doesn't work for us,
      because we have used automodule else (other than autosummary) in our docs, and to avoid duplicate index,
      we have to set ``:noindex:`` in autosummary template (see docs/templates/autosummary/module.rst).
      This breaks most of the links, where they fail to link to generated module page by using index.
      We here update the python domain role, to a general domain role (``:doc:``), and link to the page directly.
    """

    find_autosummary_original = None
    get_table_original = None

    def restore(self, *args, **kwargs):
        assert self.find_autosummary_original is not None and self.get_table_original is not None
        sphinx.ext.autosummary.generate.find_autosummary_in_files = self.find_autosummary_original
        sphinx.ext.autosummary.Autosummary.get_table = self.get_table_original

    def patch(self, app, config):
        from sphinx.ext.autosummary import Autosummary
        from sphinx.ext.autosummary.generate import AutosummaryEntry

        self.find_autosummary_original = sphinx.ext.autosummary.generate.find_autosummary_in_files
        self.get_table_original = Autosummary.get_table

        def find_autosummary_in_files(filenames: List[str]) -> List[AutosummaryEntry]:
            items: List[AutosummaryEntry] = self.find_autosummary_original(filenames)
            items = [item for item in items if item.name not in config.autosummary_mock_imports]
            return items

        def get_table(autosummary, items: List[Tuple[str, str, str, str]]) -> List[Node]:
            col_spec, autosummary_table = self.get_table_original(autosummary, items)
            if 'toctree' in autosummary.options:
                # probably within modules
                table = autosummary_table[0]
                tgroup = table[0]
                tbody = tgroup[-1]
                for row in tbody:
                    entry = row[0]
                    paragraph = entry[0]
                    pending_xref = paragraph[0]

                    # get the reference path and check whether it has been generated
                    # if path to reference is changed, this should also be changed
                    reftarget_path = 'reference/_modules/' + pending_xref['reftarget']

                    if reftarget_path in autosummary.env.found_docs:
                        # make :py:obj:`xxx` looks like a :doc:`xxx`
                        pending_xref['refdomain'] = 'std'
                        pending_xref['reftype'] = 'doc'
                        pending_xref['refexplicit'] = False
                        pending_xref['refwarn'] = True
                        pending_xref['reftarget'] = '/' + reftarget_path
                        # a special tag to enable `ResolveDocPatch`
                        pending_xref['refkeepformat'] = True

            return [col_spec, autosummary_table]

        sphinx.ext.autosummary.generate.find_autosummary_in_files = find_autosummary_in_files
        sphinx.ext.autosummary.Autosummary.get_table = get_table


class ResolveDocPatch:
    """Original :doc: role throws away all the format, and keep raw text only.
    We wish to keep module names literal. This patch is to keep literal format in :doc: resolver."""

    original = None

    def restore(self, *args, **kwargs):
        assert self.original is not None
        sphinx.domains.std.StandardDomain._resolve_doc_xref = self.original

    def patch(self, *args, **kwargs):
        self.original = sphinx.domains.std.StandardDomain._resolve_doc_xref

        def doc_xref_resolver(std_domain, env, fromdocname, builder, typ, target, node, contnode):
            if not node.get('refkeepformat'):
                # redirect to original implementation to make it safer
                return self.original(std_domain, env, fromdocname, builder, typ, target, node, contnode)

            # directly reference to document by source name; can be absolute or relative
            from sphinx.domains.std import docname_join, make_refnode
            refdoc = node.get('refdoc', fromdocname)
            docname = docname_join(refdoc, node['reftarget'])
            if docname not in env.all_docs:
                return None
            else:
                innernode = node[0]  # no astext here, to keep literal intact
                return make_refnode(builder, fromdocname, docname, None, innernode)

        sphinx.domains.std.StandardDomain._resolve_doc_xref = doc_xref_resolver


def setup(app):
    # See life-cycle of sphinx app here:
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    patch = ClassNewBlacklistPatch()
    app.connect('env-before-read-docs', patch.patch)
    app.connect('env-merge-info', patch.restore)

    patch = ResolveDocPatch()
    app.connect('env-before-read-docs', patch.patch)
    app.connect('env-merge-info', patch.restore)

    app.connect('env-before-read-docs', disable_trace_patch)

    # autosummary generate happens at builder-inited
    app.connect('config-inited', trial_tool_import_patch)

    autosummary_patch = AutoSummaryPatch()
    app.connect('config-inited', autosummary_patch.patch)
    app.connect('env-merge-info', autosummary_patch.restore)
