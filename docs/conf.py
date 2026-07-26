"""Sphinx configuration for the project documentation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from scripts.generate_documentation_evidence import generate  # noqa: E402

generate()

project = "Kinematics-Driven 316L Strain Reconstruction"
author = "Adil Kılınç et al."
copyright = "2026, Adil Kılınç et al."
version = "0.1"
release = "0.1.0"
language = "en"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_rtd_theme",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "adr/*.md",
    "from_dic_to_reconstruction.md",
    "legacy_data_contract.md",
    "mfront.md",
    "numerical_model.md",
    "partitioning.md",
    "performance.md",
    "reduced_example.md",
    "scientific_contract.md",
    "validation.md",
    "archive/**",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
}

templates_path = ["_templates"]
html_theme = "sphinx_rtd_theme"
html_title = project
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_theme_options = {
    "collapse_navigation": False,
    "includehidden": True,
    "logo_only": False,
    "navigation_depth": 2,
    "prev_next_buttons_location": "bottom",
    "sticky_navigation": True,
    "style_external_links": True,
    "style_nav_header_background": "#17324d",
    "titles_only": True,
}
html_context = {
    "display_github": True,
    "github_user": "jeffwitz",
    "github_repo": "kinematics-driven-316l-strain-reconstruction",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

graphviz_output_format = "svg"

latex_engine = "lualatex"
latex_show_urls = "footnote"
latex_use_xindy = False
latex_documents = [
    (
        root_doc,
        "kinematics-driven-316l-strain-reconstruction.tex",
        project,
        author,
        "manual",
    )
]
latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": r"""
\usepackage{unicode-math}
\setmathfont{Latin Modern Math}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{siunitx}
\sisetup{locale = US}
""",
    "sphinxsetup": "verbatimwithframe=false,verbatimhintsturnover=false",
}
