import os, sys
sys.path.insert(0, os.path.abspath('../..'))

project   = 'Smart Parking'
copyright = '2026, Adama Basse'
author    = 'Adama Basse'
release   = '1.0'
language  = 'fr'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'myst_parser',
]

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'display_version'     : True,
    'navigation_depth'    : 4,
    'titles_only'         : False,
}

# Lien "Éditer sur GitHub"
html_context = {
    'display_github' : True,
    'github_user'    : 'adama-basse',
    'github_repo'    : 'Smart_Parking',
    'github_version' : 'main',
    'conf_py_path'   : '/docs/source/',
}