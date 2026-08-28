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

# Historical pages are kept for provenance and linked from canonical task
# pages, but are intentionally not primary navigation entries. The manifest
# and structure checker enforce that distinction; Sphinx should not turn these
# documented legacy pages into build failures.
suppress_warnings = ["toc.not_included"]

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Each project lists the remote inventory first and a committed copy second.
# Intersphinx only warns when *every* location for a project fails, so the local
# copy keeps a strict `-W` build green when the remote is briefly unavailable --
# numpy.org returned a 502 on 2026-08-03 and failed the documentation job on
# main, on a commit whose own branch had passed minutes earlier.
#
# The warning carries no `type`/`subtype`, so `suppress_warnings` cannot reach
# it; a fallback location is the mechanism Sphinx actually provides for this.
#
# The remote stays first, so references still resolve against current upstream
# documentation and the copies are only a safety net. Refresh them with
# `python scripts/refresh_intersphinx_inventories.py` when they drift.
_INVENTORIES = Path(__file__).parent / "_inventories"
intersphinx_mapping = {
    "numpy": (
        "https://numpy.org/doc/stable/",
        (None, str(_INVENTORIES / "numpy.inv")),
    ),
    "python": (
        "https://docs.python.org/3/",
        (None, str(_INVENTORIES / "python.inv")),
    ),
}

# These publishers return HTTP 403 to automated HEAD/GET requests although the
# registered DOI links resolve correctly in a browser. Keep linkcheck strict
# for every other URL and ignore only the access-protected publisher families.
linkcheck_ignore = [
    r"https://doi\.org/10\.(1002|1115|1126|1287)/.*",
    r"https://onlinelibrary\.wiley\.com/doi/.*",
]

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
    "navigation_depth": 3,
    "prev_next_buttons_location": "bottom",
    "sticky_navigation": True,
    "style_external_links": True,
    "style_nav_header_background": "#17324d",
    "titles_only": False,
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
