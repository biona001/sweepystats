# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'sweepystats'
copyright = '2024, Benjamin Chu'
author = 'Benjamin Chu'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx_design",
    "numpydoc",
    "nbsphinx",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
nbsphinx_execute = 'never'  # Options: 'never', 'always', 'auto'

# allows parsing of markdown files
source_suffix = {
    '.ipynb': 'nbsphinx',
    '.rst': 'restructuredtext',
}
# allow latex rendering
myst_enable_extensions = [
    "amsmath",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    # "logo": {
    #     "image_light": "../logos/adelie-penguin.svg",
    #     "image_dark": "../logos/adelie-penguin-dark.svg",
    # },
    "github_url": "https://github.com/biona001/sweepystats",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "collapse_navigation": False,
}
html_context = {"default_mode": "bright"}
html_static_path = ['_static']
html_css_files = ["numpy.css"]
