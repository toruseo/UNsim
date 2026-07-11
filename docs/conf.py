# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "UNsim"
author = "Toru Seo"
copyright = "2026, Toru Seo"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "nbsphinx_link",
]

# Use outputs stored in the notebook; do not execute at build time
nbsphinx_execute = "never"

myst_enable_extensions = [
    "html_image",
    "dollarmath",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"

html_static_path = ["_static"]

# Google Analytics
html_js_files = [
    ("https://www.googletagmanager.com/gtag/js?id=UA-111714220-1", {"async": "async"}),
    "ga.js",
]
