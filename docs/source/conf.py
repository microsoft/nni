# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import re
import subprocess
import sys
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../extension'))


# -- Project information ---------------------------------------------------

from datetime import datetime
project = 'NNI'
copyright = f'{datetime.now().year}, Microsoft'
author = 'Microsoft'

# The short X.Y version
version = ''
# The full version, including alpha/beta/rc tags
# FIXME: this should be written somewhere globally
release = 'v3.0rc1'

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinxarg4nni.ext',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinxcontrib.bibtex',
    'sphinxcontrib.youtube',
    # 'nbsphinx',  # nbsphinx has conflicts with sphinx-gallery.
    'sphinx.ext.extlinks',
    'IPython.sphinxext.ipython_console_highlighting',
    'sphinx_tabs.tabs',
    'sphinx_copybutton',
    'sphinx_comments',

    # Custom extensions in extension/ folder.
    'tutorial_links',  # this has to be after sphinx-gallery
    'getpartialtext',
    'inplace_translation',
    'cardlinkitem',
    'codesnippetcard',
    'patch_autodoc',
    'toctree_check',
]

# Autosummary related settings
autosummary_imported_members = True
autosummary_ignore_module_all = False

# Auto-generate stub files before building docs
autosummary_generate = True

# Add mock modules
autodoc_mock_imports = [
    'apex', 'nni_node', 'tensorrt', 'pycuda', 'nn_meter', 'azureml',
    'ConfigSpace', 'ConfigSpaceNNI', 'smac', 'statsmodels', 'pybnn',
]

# Some of our modules cannot generate summary
autosummary_mock_imports = [
    'nni.retiarii.codegen.tensorflow',
    'nni.nas.benchmark.nasbench101.db_gen',
    'nni.tools.jupyter_extension.management',
] + autodoc_mock_imports

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_inherit_docstrings = False

# Sphinx will warn about all references where the target cannot be found.
nitpicky = False  # disabled for now

# A list of regular expressions that match URIs that should not be checked.
linkcheck_ignore = [
    r'http://localhost:\d+',
    r'.*://.*/#/',                           # Modern websites that has URLs like xxx.com/#/guide
    r'https://github\.com/JSong-Jia/Pic/',   # Community links can't be found any more

    # Some URLs that often fail
    r'https://www\.cs\.toronto\.edu/',                      # CIFAR-10
    r'https://help\.aliyun\.com/document_detail/\d+\.html', # Aliyun
    r'http://www\.image-net\.org/',                         # ImageNet
    r'https://www\.msra\.cn/',                              # MSRA
    r'https://1drv\.ms/',                                   # OneDrive (shortcut)
    r'https://onedrive\.live\.com/',                        # OneDrive
    r'https://www\.openml\.org/',                           # OpenML
    r'https://ml\.informatik\.uni-freiburg\.de/',
    r'https://docs\.nvidia\.com/deeplearning/',
    r'https://cla\.opensource\.microsoft\.com',
    r'https://www\.docker\.com/',
    r'https://nlp.stanford.edu/projects/glove/',

    # remove after 3.0 release
    r'https://nni\.readthedocs\.io/en/v2\.10/compression/overview\.html',
    
    r'https://github.com/google-research/google-research/blob/20736344/tunas/rematlib/mobile_model_v3.py#L453',
    r'https://github.com/google-research/google-research/blob/20736344591f774f4b1570af64624ed1e18d2867/tunas/mobile_search_space_v3.py#L728',
    r'https://github.com/quark0/darts/blob/f276dd346a09ae3160f8e3aca5c7b193fda1da37/cnn/model_search.py#L135',
    r'https://github.com/rwightman/pytorch-image-models/blob/b7cb8d03/timm/models/efficientnet_blocks.py#L134',
]

# Ignore all links located in release.rst
linkcheck_exclude_documents = ['^release']

# Bibliography files
bibtex_bibfiles = ['refs.bib']

# Add a heading to bibliography
bibtex_footbibliography_header = '.. rubric:: Bibliography'

# Set bibliography style
bibtex_default_style = 'plain'

# Sphinx gallery examples
sphinx_gallery_conf = {
    'examples_dirs': '../../examples/tutorials',   # path to your example scripts
    'gallery_dirs': 'tutorials',                   # path to where to save gallery generated output

    # Control ignored python files.
    'ignore_pattern': r'__init__\.py|/scripts/',

    # This is `/plot` by default. Only files starting with `/plot` will be executed.
    # All files should be executed in our case.
    'filename_pattern': r'.*',

    # Disabling download button of all scripts
    'download_all_examples': False,

    # Change default thumbnail
    # Working directory is strange, needs full path.
    'default_thumb_file': os.path.join(os.path.dirname(__file__), '../img/thumbnails/nni_icon_blue.png'),
}

# Copybutton: strip and configure input prompts for code cells.
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Copybutton: customize selector to exclude gallery outputs.
copybutton_selector = ":not(div.sphx-glr-script-out) > div.highlight pre"

# Allow additional builders to be considered compatible.
sphinx_tabs_valid_builders = ['linkcheck']

# Disallow the sphinx tabs css from loading.
sphinx_tabs_disable_css_loading = True

# Some tutorials might need to appear more than once in toc.
# In this list, we make source/target tutorial pairs.
# Each "source" tutorial rst will be copied to "target" tutorials.
# The anchors will be replaced to avoid dupilcate labels.
# Target should start with ``cp_`` to be properly ignored in git.
tutorials_copy_list = [
    # Seems that we don't need it for now.
    # Add tuples back if we need it in future.
]

# Toctree ensures that toctree docs do not contain any other contents.
# Home page should be an exception.
toctree_check_whitelist = [
    'index',

    # FIXME: Other exceptions should be correctly handled.
    'compression/index',
    'compression/pruning',
    'compression/quantization',
    'hpo/hpo_benchmark',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst']

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# Translation related settings
locale_dir = ['locales']

# Documents that requires translation: https://github.com/microsoft/nni/issues/4298
gettext_documents = [
    r'^index$',
    r'^quickstart$',
    r'^installation$',
    r'^(hpo|compression)/overview$',
    r'^tutorials/(pruning_quick_start_mnist|hpo_quickstart_pytorch/main)$',
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    '**.ipynb_checkpoints',
    # Exclude translations. They will be added back via replacement later if language is set.
    '**_zh.rst',
    # Exclude generated tutorials index
    'tutorials/index.rst',
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = None

# -- Options for HTML output -------------------------------------------------

# HTML logo
html_logo = '../img/nni_icon.svg'

# HTML favicon
html_favicon = '../img/favicon.ico'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_material'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {

    # Set the name of the project to appear in the navigation.
    'nav_title': 'Neural Network Intelligence',

    # Set you GA account ID to enable tracking
    'google_analytics_account': 'UA-136029994-1',

    # Specify a base_url used to generate sitemap.xml. If not
    # specified, then no sitemap will be built.
    'base_url': 'https://nni.readthedocs.io/',

    # Set the color and the accent color
    # Remember to update static/css/material_custom.css when this is updated.
    # Set those colors in layout.html.
    'color_primary': 'custom',
    'color_accent': 'custom',

    # Set the repo location to get a badge with stats
    'repo_url': 'https://github.com/microsoft/nni/',
    'repo_name': 'GitHub',

    # Visible levels of the global TOC; -1 means unlimited
    'globaltoc_depth': 5,

    # Expand all toc so that they can be dynamically collapsed
    'globaltoc_collapse': False,

    'version_dropdown': True,
    # This is a placeholder, which should be replaced later.
    'version_info': {
        'current': '/'
    },

    # Text to appear at the top of the home page in a "hero" div.
    'heroes': {
        'index': 'An open source AutoML toolkit for hyperparameter optimization, neural architecture search, '
                 'model compression and feature engineering.'
    }
}

# Disable show source link.
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
html_sidebars = {
    "**": ["logo-text.html", "globaltoc.html", "localtoc.html", "searchbox.html"]
}

html_title = 'Neural Network Intelligence'

# Add extra css files and js files
html_css_files = [
    'css/material_theme.css',
    'css/material_custom.css',
    'css/material_dropdown.css',
    'css/sphinx_gallery.css',
    'css/index_page.css',
]
html_js_files = [
    'js/version.js',
    'js/github.js',
    'js/sphinx_gallery.js',
    'js/misc.js'
]

# HTML context that can be used in jinja templates
git_commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()

html_context = {
    'git_commit_id': git_commit_id
}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'NeuralNetworkIntelligencedoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'NeuralNetworkIntelligence.tex', 'Neural Network Intelligence Documentation',
     'Microsoft', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'neuralnetworkintelligence', 'Neural Network Intelligence Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'NeuralNetworkIntelligence', 'Neural Network Intelligence Documentation',
     author, 'NeuralNetworkIntelligence', 'One line description of project.',
     'Miscellaneous'),
]


# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ['search.html']

# external links (for github code)
# Reference the code via :githublink:`path/to/your/example/code.py`
extlinks = {
    'githublink': ('https://github.com/microsoft/nni/blob/' + git_commit_id + '/%s', 'Github link: %s')
}
