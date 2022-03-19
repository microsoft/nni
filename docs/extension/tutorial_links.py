"""Creating hard links for tutorials in each individual topics."""

import os
import re


cp_list = {
    'tutorials/hello_nas.rst': 'tutorials/cp_hello_nas_quickstart.rst',
    'tutorials/pruning_quick_start_mnist.rst': 'tutorials/cp_pruning_quick_start_mnist.rst',
    'tutorials/pruning_speed_up.rst': 'tutorials/cp_pruning_speed_up.rst',
    'tutorials/quantization_quick_start_mnist.rst': 'tutorials/cp_quantization_quick_start_mnist.rst',
    'tutorials/quantization_speed_up.rst': 'tutorials/cp_quantization_speed_up.rst',
}

HEADER = """.. THIS FILE IS A COPY OF {} WITH MODIFICATIONS.
.. TO MAKE ONE TUTORIAL APPEAR IN MULTIPLE PLACES.

"""

def copy_tutorials(app):
    # TODO: use sphinx logger
    print('[tutorial links] copy tutorials...')
    for src, tar in cp_list.items():
        target_path = os.path.join(app.srcdir, tar)
        content = open(os.path.join(app.srcdir, src)).read()

        # Add a header
        content = HEADER.format(src) + content

        # Add a prefix to labels to avoid duplicates.
        label_map = {}
        for prefix, label_name in list(re.findall(r'(\.\.\s*_)(.*?)\:', content)):
            label_map[label_name] = 'tutorial_cp_' + label_name
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
