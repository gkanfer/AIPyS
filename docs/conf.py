# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import sphinx_rtd_theme
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AIPyS'
copyright = '2024, Gil Kanfer'
author = 'Gil Kanfer'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']

exclude_patterns = [
    'build',
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    'AIPyS/supportFunctions/*',
    'AIPyS/temp_function/*',
    'AIPyS/standAlone/*',
    'AIPyS/CLI/*',
    'AIPyS/temp_function/*'
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
