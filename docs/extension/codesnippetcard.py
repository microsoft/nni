"""
Code snippet card, used in index page.
"""

from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes

from sphinx.addnodes import pending_xref


CARD_TEMPLATE_HEADER = """
.. raw:: html

    <div class="codesnippet-card admonition">

    <div class="codesnippet-card-body">

    <div class="codesnippet-card-title-container">

    <div class="codesnippet-card-icon">

.. image:: {icon}

.. raw:: html

    </div>

    <h4>{title}</h4>
    </div>

"""

CARD_TEMPLATE_FOOTER = """
.. raw:: html

    </div>
"""

CARD_TEMPLATE_LINK_CONTAINER_HEADER = """
.. raw:: html

    <div class="codesnippet-card-footer">
"""

CARD_TEMPLATE_LINK = """
.. raw:: html

    <div class="codesnippet-card-link">
    {seemore}
    <span class="material-icons right">arrow_forward</span>
    </div>
"""


class CodeSnippetCardDirective(Directive):
    option_spec = {
        'icon': directives.unchanged,
        'title': directives.unchanged,
        'link': directives.unchanged,
        'seemore': directives.unchanged,
    }

    has_content = True

    def run(self):
        anchor_node = nodes.paragraph()

        try:
            title = self.options['title']
            link = directives.uri(self.options['link'])
            icon = directives.uri(self.options['icon'])
            seemore = self.options.get('seemore', 'For a full tutorial, please go here.')
        except ValueError as e:
            print(e)
            raise

        # header, title, icon...
        card_rst = CARD_TEMPLATE_HEADER.format(title=title, icon=icon)
        card_list = StringList(card_rst.split('\n'))
        self.state.nested_parse(card_list, self.content_offset, anchor_node)

        # code snippet
        self.state.nested_parse(self.content, self.content_offset, anchor_node)

        # close body
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)

        # start footer
        self.state.nested_parse(StringList(CARD_TEMPLATE_LINK_CONTAINER_HEADER.split('\n')), self.content_offset, anchor_node)

        # full tutorial link
        link_node = pending_xref(CARD_TEMPLATE_LINK,
                                 reftype='doc',
                                 refdomain='std',
                                 reftarget=link,
                                 refexplicit=False,
                                 refwarn=True,
                                 refkeepformat=True)
        # refkeepformat is handled in `patch_autodoc.py`
        self.state.nested_parse(StringList(CARD_TEMPLATE_LINK.format(seemore=seemore).split('\n')), self.content_offset, link_node)
        anchor_node += link_node

        # close footer
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)

        # close whole
        self.state.nested_parse(StringList(CARD_TEMPLATE_FOOTER.split('\n')), self.content_offset, anchor_node)

        return [anchor_node]


def setup(app):
    app.add_directive('codesnippetcard', CodeSnippetCardDirective)
