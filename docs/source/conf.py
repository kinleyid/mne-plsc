# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mne-plsc'
copyright = '2026, Isaac Kinley'
author = 'Isaac Kinley'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	'sphinx.ext.autodoc',
	'sphinx.ext.napoleon',
	'nbsphinx',
	'sphinx.ext.intersphinx'
]

napoleon_numpy_docstring = True
napoleon_use_ivar = False

# nbsphinx_execute = "never"
nbsphinx_execute = "always"
exclude_patterns = ["build", "**.ipynb_checkpoints"]

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

templates_path = ['_templates']
exclude_patterns = []

autodoc_member_order = 'groupwise'

intersphinx_mapping = {
    'pyplsc': ('https://pyplsc.readthedocs.io/en/stable/', None),
    'mne': ('https://mne.tools/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

nbsphinx_allow_errors = True

# Set up for static rendering of surface images
import pyvista as pv
from mne.viz._brain import _BrainScraper as BrainScraper
pv.set_plot_theme("document")
pv.OFF_SCREEN = True
os.environ["_MNE_BUILDING_DOC"] = "true"
sphinx_gallery_conf = {
    "image_scrapers": ("matplotlib", BrainScraper, "pyvista")
}