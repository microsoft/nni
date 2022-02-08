import os

from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from docutils import nodes

CARD_TEMPLATE = """
.. raw:: html

    <div class="card-link admonition">

    <a href="{link}">

    <div class="card-link-body">

    <div class="card-link-title-container">
        <h4>{header}</h4>
    </div>

    <p class="card-link-summary">{description}</p>

    <div class="card-link-icon circle">

.. image:: {image}

.. raw:: html

    </div>

    </div>

    </a>

    </div>
"""


class CustomCardItemDirective(Directive):
    option_spec = {
        'header': directives.unchanged,
        'image': directives.unchanged,
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
                image = os.path.join(os.path.relpath(env.srcdir, env.confdir), '../img/thumbnails/nni_icon_blue.png')

            if 'description' in self.options:
                description = self.options['description']
            else:
                description = ''

            if 'tags' in self.options:
                tags = self.options['tags']
            else:
                tags = ''

        except FileNotFoundError as e:
            print(e)
            return []
        except ValueError as e:
            print(e)
            raise

        card_rst = CARD_TEMPLATE.format(header=header,
                                        image=image,
                                        link=link,
                                        description=description,
                                        tags=tags)
        card_list = StringList(card_rst.split('\n'))
        card = nodes.paragraph()
        self.state.nested_parse(card_list, self.content_offset, card)
        return [card]


def setup(app):
    app.add_directive('cardlinkitem', CustomCardItemDirective)
