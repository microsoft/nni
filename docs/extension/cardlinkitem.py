"""
Directive "cardlinkitem" used in tutorials navigation page.
"""

import os

from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes
from sphinx.addnodes import pending_xref

TAG_TEMPLATE = """<span class="card-link-tag">{tag}</span>"""

TAGS_TEMPLATE = """
    <p class="card-link-summary">{tags}</p>
"""

CARD_HEADER = """
.. raw:: html

    <div class="card-link admonition">

    <a class="card-link-clickable" href="#">

    <div class="card-link-body">

    <div class="card-link-text">

    <div class="card-link-title-container">
        <h4>{header}</h4>
    </div>

    <p class="card-link-summary">{description}</p>

    {tags}

    </div>

    <div class="card-link-icon circle {image_background}">

.. image:: {image}

.. raw:: html

    </div>

    </div>

    </a>
"""

CARD_FOOTER = """
.. raw:: html

    </div>
"""


class CustomCardItemDirective(Directive):
    option_spec = {
        'header': directives.unchanged,
        'image': directives.unchanged,
        'background': directives.unchanged,
        'link': directives.unchanged,
        'description': directives.unchanged,
        'tags': directives.unchanged
    }

    def run(self):
        env = self.state.document.settings.env

        try:
            if 'header' in self.options:
                header = self.options['header']
            else:
                raise ValueError('header not found')

            if 'link' in self.options:
                link = directives.uri(self.options['link'])
            else:
                raise ValueError('link not found')

            if 'image' in self.options:
                image = directives.uri(self.options['image'])
            else:
                image = os.path.join(os.path.relpath(env.app.srcdir, env.app.confdir), '../img/thumbnails/nni_icon_white.png')

            image_background = self.options.get('background', 'indigo')
            description = self.options.get('description', '')

            tags = self.options.get('tags', '').strip().split('/')
            tags = [t.strip() for t in tags if t.strip()]

        except ValueError as e:
            print(e)
            raise

        if tags:
            tags_rst = TAGS_TEMPLATE.format(tags=''.join([TAG_TEMPLATE.format(tag=tag) for tag in tags]))
        else:
            tags_rst = ''

        card_rst = CARD_HEADER.format(
            header=header,
            image=image,
            image_background=image_background,
            link=link,
            description=description,
            tags=tags_rst)
        card = nodes.paragraph()
        self.state.nested_parse(StringList(card_rst.split('\n')), self.content_offset, card)

        # This needs to corporate with javascript: propagate_card_link.
        # because sphinx can't correctly handle image in a `pending_xref` after `keepformat`.
        link_node = pending_xref('<a/>',
                                 reftype='doc',
                                 refdomain='std',
                                 reftarget=link,
                                 refexplicit=False,
                                 refwarn=True)
        link_node += nodes.paragraph(header)
        link_node['classes'] = ['card-link-anchor']
        card += link_node

        self.state.nested_parse(StringList(CARD_FOOTER.split('\n')), self.content_offset, card)

        return [card]


def setup(app):
    app.add_directive('cardlinkitem', CustomCardItemDirective)
