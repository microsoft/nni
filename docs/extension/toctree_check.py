"""
Make sure pages that contain toctree only has a toctree,
because, if our theme is used, other contents will not be visible.
"""

import re

from docutils import nodes
from sphinx.application import Sphinx
from sphinx.addnodes import toctree
from sphinx.util.logging import getLogger

logger = getLogger('toctree_check')

def _strip_compound(node):
    if isinstance(node, nodes.compound):
        return _strip_compound(node[0])
    return node


def toctree_check(app: Sphinx, doctree: nodes.document, docname: str):
    whitelist = app.config.toctree_check_whitelist

    if docname in whitelist:
        return

    # Scan top-level nodes
    has_toctree = False

    other_types = []

    for i in range(len(doctree[0])):
        node = doctree[0][i]
        if isinstance(_strip_compound(node), toctree):
            has_toctree = True
        elif isinstance(_strip_compound(node), nodes.title):
            # Allow title
            pass
        else:
            other_types.append(type(_strip_compound(node)))

    if has_toctree and other_types:
        # We don't allow a document with toctree to have other types of contents
        logger.warning('Expect a toctree document to contain only a toctree, '
                       'but found other types of contents: %s', str(set(other_types)),
                       location=docname)


def setup(app):
    app.connect('doctree-resolved', toctree_check)

    app.add_config_value('toctree_check_whitelist', [], True)
