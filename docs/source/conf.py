# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import qexpy

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "qexpy"
copyright = "2025, Astral Cai, Ryan Martin, Connor Kapahi"
author = "Astral Cai, Ryan Martin, Connor Kapahi"
release = qexpy.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
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
html_css_files = ["custom.css"]
html_copy_source = False
html_theme_options = {
    "navbar_align": "left",
    "collapse_navigation": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Queens-Physics/qexpy",
            "icon": "fa-brands fa-github",
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
}
html_context = {"default_mode": "light"}
html_sidebars = {"**": ["sidebar-nav-bs.html"]}

# -- Options for autodoc -----------------------------------------------------

autodoc_typehints = "none"
autodoc_member_order = "bysource"
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False
numpydoc_show_class_members = False
