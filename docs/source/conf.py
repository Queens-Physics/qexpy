# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import re
import sys

sys.path.insert(0, os.path.abspath("../.."))

import qexpy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "QExPy"
copyright = "2024, Astral Cai"
author = "Astral Cai"
release = qexpy.__version__
version = re.match(r"^(\d+\.\d+)", release).expand(r"\1")
language = "en"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "nbsphinx",
    "sphinx_design",
    "numpydoc",
    "IPython.sphinxext.ipython_console_highlighting",
    "IPython.sphinxext.ipython_directive",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for Linking -----------------------------------------------------

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo-colour.svg"
html_favicon = "_static/favicon-colour.svg"
html_static_path = ["_static"]
html_css_files = ["qexpy.css"]
html_copy_source = False

html_theme_options = {
    "navbar_end": ["navbar-icon-links"],
    "collapse_navigation": True,
    "navigation_depth": 4,
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Queens-Physics/qexpy",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
        {
            "name": "Queen's Physics",
            "url": "https://www.queensu.ca/physics/",
            "icon": "_static/index-images/queens-logo.png",
            "type": "local",
        },
    ],
    "primary_sidebar_end": [],
    "navigation_with_keys": True,
}


html_context = {"default_mode": "light"}

html_sidebars = {"**": ["sidebar-nav-bs.html"]}


# -- Options for autodoc -----------------------------------------------------

autosummary_generate = True
autodoc_typehints = "none"

# -- Options for Numpydoc -----------------------------------------------------

numpydoc_attributes_as_param_list = False
