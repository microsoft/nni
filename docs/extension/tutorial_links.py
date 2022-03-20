"""Creating hard links for tutorials in each individual topics."""

import os
import re

HEADER = """.. THIS FILE IS A COPY OF {} WITH MODIFICATIONS.
.. TO MAKE ONE TUTORIAL APPEAR IN MULTIPLE PLACES.

"""

def flatten_filename(filename):
    return filename.replace('/', '_').replace('.', '_')

def copy_tutorials(app):
    # TODO: use sphinx logger
    print('[tutorial links] copy tutorials...')
    for src, tar in app.config.tutorials_copy_list:
        target_path = os.path.join(app.srcdir, tar)
        content = open(os.path.join(app.srcdir, src)).read()

        # Add a header
        content = HEADER.format(src) + content

        # Add a prefix to labels to avoid duplicates.
        label_map = {}

        # find all anchors:   https://www.sphinx-doc.org/en/master/usage/restructuredtext/roles.html
        # but not hyperlinks: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#external-links
        for prefix, label_name in list(re.findall(r'(\.\.\s*_)(.*?)\:\s*\n', content)):
            label_map[label_name] = flatten_filename(tar) + '_' + label_name
            # anchor
            content = content.replace(prefix + label_name + ':', prefix + label_map[label_name] + ':')
            # :ref:`xxx`
            content = content.replace(f':ref:`{label_name}`', f':ref:`{label_map[label_name]}')
            # :ref:`yyy <xxx>`
            content = re.sub(r"(\:ref\:`.*?\<)" + label_name + r"(\>`)", r'\1' + label_map[label_name] + r'\2', content)

        open(target_path, 'w').write(content)


def setup(app):
    # See life-cycle of sphinx app here:
    # https://www.sphinx-doc.org/en/master/extdev/appapi.html#sphinx-core-events
    app.connect('builder-inited', copy_tutorials)
    app.add_config_value('tutorials_copy_list', [], True, [list])
