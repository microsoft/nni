"""
Sphinx inplace translation.
"""

__version_info__ = (1, 0, 0)
__version__ = '1.0.0'

import os
import types


def builder_inited(app):
    """Event listener to set up multiple environments."""
    patch_doc2path(app.env, app.config.language)


def patch_doc2path(env, language):
    # patch doc2path so that it resolves to the correct language.
    override_doc2path = env.doc2path

    def doc2path(env, docname: str, base: bool = True):
        path = override_doc2path(docname, base)
        if language not in (None, 'en'):
            # The language is set to another one
            new_docname = f'{docname}_{language}'
            new_path = override_doc2path(new_docname, base)
            if os.path.exists(new_path):
                return new_path
        return path

    env.doc2path = types.MethodType(doc2path, env)


def setup(app):
    app.connect('builder-inited', builder_inited)
    # app.connect('source-read', render_jinja)
    return {
        'version': __version__,
    }
