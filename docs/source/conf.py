# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'gtrace'
copyright = '2019-2026, Yoichi Aso'
author = 'Yoichi Aso'

# The full version, including alpha/beta/rc tags. Keep in step with
# setup.py and gtrace.__version__.
release = '0.3.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['autoapi.extension','sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.mathjax',
    "nbsphinx"
]

# Render the "Attributes" section of a numpydoc docstring as :ivar: fields
# rather than as separate attribute directives. autoapi already emits a
# directive for every real class attribute, so without this every attribute
# that is also documented in the docstring is defined twice, which Sphinx
# reports as a duplicate object description and which makes the resulting
# cross references ambiguous.
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
#
# 'api' holds hand-written automodule stubs that predate sphinx-autoapi.
# autoapi now generates the whole API reference from the source, so
# building both documents every object twice: Sphinx reports several
# hundred "duplicate object description" warnings and every cross
# reference becomes ambiguous. The stubs are kept for now but not built.
exclude_patterns = ['api']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

autoapi_dirs = ['../../gtrace']

# The default list minus 'imported-members'. gtrace re-exports heavily
# (``from gtrace.draw.draw import *``, ``from numpy import array``), and
# documenting the imports as well means the same object is described
# twice under two names. Sphinx then cannot resolve a reference such as
# ``draw.Canvas``, because both gtrace.draw.Canvas and
# gtrace.draw.draw.Canvas match it. Each object is now documented once,
# where it is defined.
autoapi_options = [
    'members',
    'undoc-members',
    'private-members',
    'show-inheritance',
    'show-module-summary',
    'special-members',
]
