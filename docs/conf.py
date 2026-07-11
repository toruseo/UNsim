# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

project = "UNsim"
author = "Toru Seo"
copyright = "2026, Toru Seo"

extensions = [
    "myst_parser",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

language = "en"

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
