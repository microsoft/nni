"""Additional docutils patch to suppress warnings in i18n documentation build."""

from typing import Any

import docutils
from docutils.utils import Reporter


class Patch:
    """
    This is actually done in sphinx, but sphinx didn't replace all `get_language` occurrences.
    https://github.com/sphinx-doc/sphinx/blob/680417a10df7e5c35c0ff65979bd22906b9a5f1e/sphinx/util/docutils.py#L127

    Related issue:
    https://github.com/sphinx-doc/sphinx/issues/10179
    """

    original = None

    def restore(self, *args, **kwargs):
        assert self.original is not None
        docutils.parsers.rst.languages.get_language = self.original

    def patch(self, *args, **kwargs):
        from docutils.parsers.rst.languages import get_language
        self.original = get_language

        def patched_get_language(language_code: str, reporter: Reporter = None) -> Any:
            return get_language(language_code)

        docutils.parsers.rst.languages.get_language = patched_get_language


def setup(app):
    # See life-cycle of sphinx app here:
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    patch = Patch()
    app.connect('env-before-read-docs', patch.patch)
    app.connect('env-merge-info', patch.restore)
