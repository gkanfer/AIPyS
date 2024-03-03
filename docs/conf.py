import os
import sys
sys.path.insert(0,os.path.join('..','AIPyS'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'aipys-sphinx-test'
copyright = '2024, Gil Kanfer'
author = 'Gil Kanfer'
release = '0.0.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = [
    # Directories
    'CLI', 
    'standAlone', 
    'supportFunctions', 
    'temp_function', 
    'web_app', 
    'build',
    # Single files
    'setup.py',
    # Other common patterns to exclude
    '_build', 
    'Thumbs.db', 
    '.DS_Store',
    ]


autodoc_mock_imports = [
   
    "scikit-image",
    ]

# autodoc_mock_imports = [
#     "pytorch",
#     "cudatoolkit",
#     "scikit-image",
#     "matplotlib",
#     "pandas",
#     "seaborn",
#     "scikit-learn",
#     "opencv",
#     "pytensors",
#     "pymc",
#     "dash",
#     "dash-core-components",
#     "dash-html-components",
#     "dash-renderer",
#     "dash-table",
#     "dash-bootstrap-components",
#     "plotly_express"
# ]
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']