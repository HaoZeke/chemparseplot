# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
project = "chemparseplot"
project_copyright = "2023-2026, Rohit Goswami"
author = "Rohit Goswami"
# The short X.Y version.
version = "1.9.11"
# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "autodoc2",  # Consumes docstrings
    "myst_nb",  # Markdown notebooks
    "sphinx.ext.napoleon",  # Allows for Google Style Docs
    "sphinx.ext.todo",  # Show TODO details
    "sphinx_copybutton",  # Let there be plagiarism!
    "sphinx_sitemap",  # Always a good idea
    "sphinx_togglebutton",  # Toggles reduce clutter
    "sphinx.ext.viewcode",  # Links to source code
    "sphinx.ext.intersphinx",  # Connects to other documentation
    "sphinx_design",  # grids, cards, tabs, dropdowns (Shibuya-friendly)
    "sphinxcontrib.mermaid",  # architecture / data-flow diagrams
]

myst_enable_extensions = [
    "fieldlist",
    "dollarmath",
    "colon_fence",  # ::: {note} style fences for design/admonitions
    "attrs_block",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    # Link to other rgpkgs packages
    "rgpycrumbs": ("https://rgpycrumbs.rgoswami.me", None),
    "pychum": ("https://pychum.rgoswami.me", None),
}
templates_path = ["_templates"]
exclude_patterns = []

# Sitemap Config
html_baseurl = "https://chemparseplot.rgoswami.me"

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

# Mermaid
mermaid_version = "11.4.0"
mermaid_init_js = "mermaid.initialize({startOnLoad:true, theme:'neutral'});"

# -- Options for HTML output -------------------------------------------------
html_theme = "shibuya"
html_static_path = ["_static"]
html_css_files = []
html_js_files = []

html_context = {
    "source_type": "github",
    "source_user": "HaoZeke",
    "source_repo": "chemparseplot",
    "source_version": "main",
    "source_docs_path": "/doc/source/",
}

html_theme_options = {
    "github_url": "https://github.com/HaoZeke/chemparseplot",
    "accent_color": "cyan",
    "dark_code": True,
    "globaltoc_expand_depth": 1,
    "toctree_collapse": True,
    "toctree_maxdepth": 3,
    "toctree_titles_only": True,
    "nav_links": [
        {
            "title": "Ecosystem",
            "children": [
                {
                    "title": "rgpycrumbs",
                    "url": "https://rgpycrumbs.rgoswami.me",
                    "summary": "CLI suite, surfaces, and eOn plot dispatch",
                    "external": True,
                },
                {
                    "title": "eOn",
                    "url": "https://eondocs.org",
                    "summary": "Long-timescale MD / NEB engine",
                    "external": True,
                },
                {
                    "title": "pychum",
                    "url": "https://pychum.rgoswami.me",
                    "summary": "ORCA / NWChem input generation",
                    "external": True,
                },
            ],
        },
        {
            "title": "PyPI",
            "url": "https://pypi.org/project/chemparseplot/",
            "external": True,
        },
    ],
}

html_sidebars = {
    "**": [
        "sidebars/localtoc.html",
        "sidebars/repo-stats.html",
        "sidebars/edit-this-page.html",
    ],
}
