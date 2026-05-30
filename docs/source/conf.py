import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project   = 'Smart Parking'
copyright = '2026, Adama Basse & Djibril Sall'
author    = 'Adama Basse'
release   = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'myst_parser',
]

templates_path   = ['_templates']
exclude_patterns = []
language         = 'fr'
html_theme       = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only'           : False,
    'navigation_depth'    : 4,
    'style_nav_header_background': '#0A1628',
}