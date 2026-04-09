project = "geoews"
author = "Alexander Sokol"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "numpydoc",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True
autodoc_typehints = "description"
numpydoc_show_class_members = False

templates_path = ["_templates"]
exclude_patterns: list[str] = []

html_theme = "sphinx_rtd_theme"
