# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "chemparseplot"
copyright = f'2023-{date.today().year}, Rohit Goswami'
author = "Rohit Goswami"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autodoc2",  # Consumes docstrings
    "myst_parser",  # Markdown documentation
    "myst_nb",  # Markdown notebooks
    "sphinx.ext.napoleon",  # Allows for Google Style Docs
    "sphinx.ext.todo",  # Show TODO details
    "sphinx_copybutton",  # Let there be plagiarism!
    "sphinx_sitemap",  # Always a good idea
    "sphinx_togglebutton",  # Toggles reduce clutter
    "sphinx.ext.viewcode",  # Links to source code
    "sphinx.ext.intersphinx",  # Connects to other documentation
]

myst_enable_extensions = [
    "fieldlist",
    "dollarmath",
]

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
templates_path = ["_templates"]
exclude_patterns = []

# Sitemap Config
html_baseurl = "https://haozeke.github.io/chemparseplot"

# MathJax Configuration
mathjax_config = {
    "extensions": ["tex2jax.js"],
    "jax": ["input/TeX", "output/HTML-CSS"],
}

# API Doc settings
autodoc2_render_plugin = "myst"
autodoc2_packages = [
    "../../chemparseplot",
]
# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "library"
# Sidebars
html_sidebars = {
    "**": [
        "about.html",  # Project name, description, etc.
        "searchbox.html",  # Search.
        "extralinks.html",  # Links specified in theme options.
        "globaltoc.html",  # Global table of contents.
        "localtoc.html",  # Contents of the current page.
        "readingmodes.html",  # Light/sepia/dark color schemes.
    ]
}

html_theme_options = {
    "show_breadcrumbs": True,
    "reading_mode": "sepia",
    "typography": "book",
    "extra_links": {
        "Github": "https://github.com/HaoZeke/chemparseplot",
        "Personal": "https://rgoswami.me",
    },
}
html_static_path = ["_static"]
