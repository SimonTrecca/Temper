# docs/sphinx/source/conf.py
import os
import sys

project = 'Temper'
author = 'Your Name'
version = '0.1'
release = version

extensions = [
    'myst_parser',        # allow Markdown
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.todo',
    'breathe',
    'exhale',
]

html_theme = "sphinx_rtd_theme"

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "deflist",
    "colon_fence",
]

todo_include_todos = True

# Breathe/Exhale
breathe_default_project = "Temper"

exhale_args = {
    "containmentFolder": os.path.join(os.path.dirname(__file__), "api"),
    "rootFileName":      "library_root.rst",
    "rootFileTitle":     "API Reference",
    "doxygenStripFromPath":  os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    ),
    "createTreeView":    True,
}

suppress_warnings = ["autosectionlabel.*"]
